import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...utils.builder import TRANSFORMER
from ...utils.net_utils import xavier_init
from ...utils.transform import inverse_sigmoid
from ...utils.logger import logger
from ...utils.misc import param_size
from ...utils.transform import bchw_2_bhwc, denormalize
from ..bricks.point_transformers import (
    ptTransformerBlock,
    ptTransformerBlock_CrossAttn,
)
from ..bricks.metro_transformer import BertConfig, METROBlock
from ..bricks.pt_metro_transformer import point_METRO_block
from lib.utils.config import CN
from ...utils.collation import generate_grid_sample_proj


@TRANSFORMER.register_module()
class PtEmbedTRv2(nn.Module):

    def __init__(self, cfg):
        super(PtEmbedTRv2, self).__init__()
        self._is_init = False

        self.nblocks = cfg.N_BLOCKS  # 6, stack the Transformer 6 times
        self.nneighbor = cfg.N_NEIGHBOR  # 16
        self.nneighbor_query = cfg.N_NEIGHBOR_QUERY  # 16
        self.nneighbor_decay = cfg.get("N_NEIGHBOR_DECAY", True)
        self.transformer_dim = cfg.TRANSFORMER_DIM  # 256
        self.feat_dim = cfg.POINTS_FEAT_DIM  # 256
        self.with_point_embed = cfg.WITH_POSI_EMBED  # True

        self.predict_inv_sigmoid = cfg.get("PREDICT_INV_SIGMOID", False)

        self.feats_self_attn = ptTransformerBlock(self.feat_dim, self.transformer_dim, self.nneighbor)
        self.query_feats_cross_attn = nn.ModuleList()
        self.query_self_attn = nn.ModuleList()

        for i in range(self.nblocks):
            self.query_self_attn.append(ptTransformerBlock(self.feat_dim, self.transformer_dim,
                                                           self.nneighbor_query))  # Self-attention block
            self.query_feats_cross_attn.append(
                ptTransformerBlock_CrossAttn(self.feat_dim,
                                             self.transformer_dim,
                                             self.nneighbor,
                                             expand_query_dim=False))  # Cross-attention block

        # self.init_weights()
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def init_weights(self):
        if self._is_init == True:
            return

        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
        logger.info(f"{type(self).__name__} init done")

    def forward(self,
                pt_xyz,
                pt_feats,
                query_xyz,
                reg_branches,
                query_feat=None,
                pt_embed=None,
                query_emb=None,
                **kwargs):
        if pt_embed is not None and self.with_point_embed:
            pt_feats = pt_feats + pt_embed

        if query_feat is None:
            query_feats = query_emb
        else:
            query_feats = query_feat + query_emb

        pt_feats, _ = self.feats_self_attn(pt_xyz, pt_feats)

        query_xyz_n = []
        query_feats_n = []

        # query_feats = query_emb
        for i in range(self.nblocks):
            query_feats, _ = self.query_self_attn[i](query_xyz, query_feats)  # self-attention

            query = torch.cat((query_xyz, query_feats), dim=-1)

            query_feats, _ = self.query_feats_cross_attn[i](pt_xyz, pt_feats, query)  # cross-attention

            # FFN
            if self.predict_inv_sigmoid:
                query_xyz = reg_branches[i](query_feats) + inverse_sigmoid(query_xyz)
                query_xyz = query_xyz.sigmoid()
            else:
                query_xyz = reg_branches[i](query_feats) + query_xyz

            # Store the intermediate results
            query_xyz_n.append(query_xyz)
            query_feats_n.append(query_feats)

        return torch.stack(query_xyz_n)


class _Sequential(nn.Sequential):
    """
        A wrapper to allow nn.Sequential to accept multiple inputs
    """

    def forward(self, query_feats, query_xyz, pt_feats, pt_xyz):
        query_xyz_n = []
        for i, module in enumerate(self._modules.values()):
            query_feats, query_xyz, pred_pose, pred_shape = module(query_xyz, query_feats, pt_xyz, pt_feats)
            query_xyz_n.append(query_xyz)
        query_xyz_n = torch.stack(query_xyz_n)
        return query_xyz_n, pred_pose, pred_shape


class MetroTR(nn.Module):

    def __init__(self, cfg):
        super(MetroTR, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        input_feat_dim = cfg.INPUT_FEAT_DIM
        hidden_feat_dim = cfg.HIDDEN_FEAT_DIM
        output_feat_dim = input_feat_dim[1:] + [3]
        self.dropout = cfg.DROP_OUT

        self.num_hidden_layers = cfg.NUM_HIDDEN_LAYERS
        self.num_attention_heads = cfg.NUM_ATTENTION_HEADS
        self.bps_feature_dim = cfg.BPS_FEAT_DIM

        # * load metro block
        self.metro_encoder = []
        self.layer_num = len(output_feat_dim)
        # init three transformer-encoder blocks in a loop
        for i in range(self.layer_num):
            config_class, model_class = BertConfig, point_METRO_block

            config = config_class.from_pretrained("lib/external/metro/bert_cfg.json")
            config.output_attentions = False
            config.hidden_dropout_prob = self.dropout
            config.img_feature_dim = input_feat_dim[i]
            config.output_feature_dim = output_feat_dim[i]
            self.hidden_size = hidden_feat_dim[i]
            self.intermediate_size = self.hidden_size * 4

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
            for _, param in enumerate(update_params):
                arg_param = getattr(self, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # Required, as default value 512 < 799/4096
            config.max_position_embeddings = 4096

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            logger.info("Init model from scratch.")
            self.metro_encoder.append(model)

        self.metro_encoder = _Sequential(*self.metro_encoder)

        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def forward(self, query_xyz, query_feat, point_xyz, point_feat):
        mesh_feat = torch.concat((query_xyz, query_feat), dim=2)
        bps_feat = torch.concat((point_xyz, point_feat), dim=2)
        pred_verts = self.metro_encoder(mesh_feat, bps_feat)
        return pred_verts


@TRANSFORMER.register_module()
class PtEmbedTRv3(nn.Module):
    """
        Implements the idea of METRO transformer -> POEM transformer
        Use the result [799, 3] of the METRO transformer in POEM transformer
        A 3 + 3 structure
    """

    def __init__(self, cfg):
        super(PtEmbedTRv3, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        # Configuration for vt_config
        self.vt_config = dict(
            INPUT_FEAT_DIM=cfg.VT_INPUT_FEAT_DIM,
            HIDDEN_FEAT_DIM=cfg.VT_HIDDEN_FEAT_DIM,
            DROP_OUT=cfg.VT_DROPOUT,
            NUM_HIDDEN_LAYERS=cfg.VT_NUM_HIDDEN_LAYERS,
            NUM_ATTENTION_HEADS=cfg.VT_NUM_ATTENTION_HEADS,
            BPS_FEAT_DIM=cfg.PT_POINTS_FEAT_DIM  # align with the dimension of Point Transformer
        )

        # Configuration for pt_config
        self.pt_config = dict(N_BLOCKS=cfg.PT_N_BLOCKS,
                              N_NEIGHBOR=cfg.PT_N_NEIGHBOR,
                              N_NEIGHBOR_QUERY=cfg.PT_N_NEIGHBOR_QUERY,
                              N_NEIGHBOR_DECAY=cfg.get("PT_N_NEIGHBOR_DECAY", True),
                              POINTS_FEAT_DIM=cfg.PT_POINTS_FEAT_DIM,
                              WITH_POSI_EMBED=cfg.PT_WITH_POSI_EMBED,
                              TRANSFORMER_DIM=cfg.PT_TRANSFORMER_DIM,
                              PREDICT_INV_SIGMOID=cfg.get("PT_PREDICT_INV_SIGMOID", False))

        self.feat_dim = cfg.PT_POINTS_FEAT_DIM
        self.embed_dims = cfg.PT_POINTS_FEAT_DIM
        self.nsample = cfg.VT_KEY_NSAMPLE
        self.metro_transformer = MetroTR(CN(self.vt_config))
        self.point_transformer = PtEmbedTRv2(CN(self.pt_config))

        # >>>>>>>>>>>>  End section
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def merge_features(self, q, merge_net, master_id):
        """
            q: [B, nsample, N, 256]
            q_merged: [B, nsample, 256]
        """
        master_is_zero = torch.sum(master_id)
        assert master_is_zero == 0, "only support master_id is 0"
        cam_num = q.shape[2]

        q1 = q[:, :, 0, :]
        q = merge_net[0](q)
        master_features = q[:, :, 0, :]  # [B, nsample, 128]
        other_features = q[:, :, 1:, :]  # [B, nsample, 7, 128]
        q = torch.matmul(other_features, master_features.unsqueeze(-1))  # [B, nsample, 7, 1]
        q = torch.matmul(other_features.transpose(2, 3), q).squeeze(-1)
        q_merged = merge_net[1](q) / cam_num  # [B, nsample, 256]
        q_merged = q1 + q_merged
        return q_merged

    def forward(self,
                pt_xyz,
                pt_feats,
                reg_branches,
                pt_embed,
                query_feat,
                query_xyz,
                img_metas,
                feature_map,
                merge_branch,
                reference_hand_center,
                radius,
                query_emb=None,
                **kwargs):

        batch_size = pt_xyz.shape[0]
        BN = np.sum(img_metas["cam_view_num"])
        self.radius = radius

        pred_verts_metro = self.metro_transformer(query_xyz, query_feat, pt_xyz, pt_feats)  # (B, 799, 3)
        # >>>>>>>>>>>>>> Grid sample the verts features and merge the multi-view features

        # rescale it back to original space first
        pred_verts_metro_raw = pred_verts_metro * self.radius + reference_hand_center.unsqueeze(1).repeat(1, 799, 1)

        # Project the points to sample the features
        pred_verts_2d = generate_grid_sample_proj(pred_verts_metro_raw, img_metas)
        pred_verts_2d = pred_verts_2d.unsqueeze(-2)  # [B, 799, 1, 2]
        ref_proj_2d = torch.einsum("bijk, k->bijk", pred_verts_2d,
                                   1.0 / img_metas["inp_res"])  # TENSOR (BN,  nsample, 2), [0 ~ 1]
        ref_proj_2d = ref_proj_2d * 2 - 1

        query_feat = F.grid_sample(feature_map, ref_proj_2d, align_corners=False)\
                    .squeeze(-1).reshape(BN, self.feat_dim, 799)

        # Similar logic as merging the features for bps_sampled_feat
        query_feats_full = []
        master_id = img_metas["master_id"]
        for i in range(batch_size):
            start_idx = np.sum(img_metas["cam_view_num"][:i])
            end_idx = np.sum(img_metas["cam_view_num"][:i + 1])

            query_feat_sub = query_feat[start_idx:end_idx].view(1, -1, img_metas["cam_view_num"][i], query_feat.size(1))
            query_feat_sub = self.merge_features(query_feat_sub, merge_branch, torch.Tensor([master_id[i]]))

            query_feats_full.append(query_feat_sub)

        query_feat = torch.concat(query_feats_full)
        # >>>>>>>>>>>>>>>> End section

        # Send pre_verts_metro (normalized) into the Point Transformer
        pred_verts_PT = self.point_transformer(pt_xyz, pt_feats, pred_verts_metro, reg_branches, query_feat, pt_embed,
                                               query_emb)
        pred_verts = torch.concat((pred_verts_metro.unsqueeze(0), pred_verts_PT), dim=0)

        return pred_verts


@TRANSFORMER.register_module()
class PtEmbedTRv4(nn.Module):

    def __init__(self, cfg):
        super(PtEmbedTRv4, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        # Configuration for METRO part
        self.input_feat_dim = cfg.INPUT_FEAT_DIM
        self.hidden_feat_dim = self.input_feat_dim
        self.output_feat_dim = self.input_feat_dim
        self.dropout = cfg.DROPOUT
        self.num_hidden_layers = cfg.NUM_HIDDEN_LAYERS
        self.num_attention_heads = cfg.NUM_ATTENTION_HEADS
        self.bps_feature_dim = cfg.BPS_FEAT_DIM
        self.parametric_output = cfg.get("PARAMETRIC_OUTPUT", False)
        self.mano_center_idx = cfg.get("TRANSFORMER_CENTER_IDX", 9)

        # Configuration for PT part
        self.nneighbor = cfg.N_NEIGHBOR
        self.nneighbor_query = cfg.N_NEIGHBOR_QUERY

        # * load metro block
        self.pt_metro_encoder = []
        self.layer_num = cfg.N_BLOCKS

        # init three transformer-encoder blocks in a loop
        for i in range(self.layer_num):
            config_class, model_class = BertConfig, point_METRO_block

            config = config_class.from_pretrained("config/backbone/bert_cfg.json")
            config.output_attentions = False
            config.hidden_dropout_prob = self.dropout
            config.img_feature_dim = self.input_feat_dim
            config.output_feature_dim = self.output_feat_dim
            config.bps_feature_dim = self.bps_feature_dim + 3
            config.parametric_output = self.parametric_output
            config.center_idx = self.mano_center_idx
            self.hidden_size = self.hidden_feat_dim
            self.intermediate_size = self.hidden_size * 4

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
            for _, param in enumerate(update_params):
                arg_param = getattr(self, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    setattr(config, param, arg_param)

            # Required, as default value 512 < 799/4096
            config.max_position_embeddings = 4096

            # Add the PT part to config
            config.n_neighbor = self.nneighbor
            config.n_neighbor_query = self.nneighbor_query
            config.init_block = True if i == 0 else False  # ! init_block won't use KNN for vec_attn
            config.final_block = True if i == self.layer_num - 1 else False  # Final block used for potential parametric output

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            self.pt_metro_encoder.append(model)

        self.pt_metro_encoder = _Sequential(*self.pt_metro_encoder)

        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def forward(self, query_xyz, query_feat, pt_xyz, pt_feats):
        pred_verts, pred_pose, pred_shape = self.pt_metro_encoder(query_feats=query_feat,
                                                                  query_xyz=query_xyz,
                                                                  pt_feats=pt_feats,
                                                                  pt_xyz=pt_xyz)
        return pred_verts, pred_pose, pred_shape
