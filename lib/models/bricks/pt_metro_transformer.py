import numpy as np
import scipy
import torch
from torch import nn
from transformers.models.bert.modeling_bert import (BertConfig, BertEmbeddings, BertEncoder, BertPooler, BertAttention,
                                                    BertIntermediate, BertOutput, BertPreTrainedModel,
                                                    apply_chunking_to_forward)
from .point_transformers import ptTransformerBlock, ptTransformerBlock_CrossAttn
from ...utils.transform import rot6d_to_aa
from manotorch.manolayer import ManoLayer


class pointer_layer(nn.Module):

    def __init__(self, config):
        super(pointer_layer, self).__init__()

        self.nneighbor = config.n_neighbor  # 32
        self.nneighbor_query = config.n_neighbor_query  # 32
        self.nneighbor_decay = True
        self.feat_dim = config.img_feature_dim  # 256
        self.transformer_dim = self.feat_dim  # 256
        self.init_block = config.init_block  # True for first block, False otherwise

        self.reg_branch = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(), nn.Linear(self.feat_dim, 3))
        self.query_self_attn = ptTransformerBlock(self.feat_dim, self.transformer_dim, self.nneighbor_query,
                                                  self.init_block)
        self.query_cross_attn = ptTransformerBlock_CrossAttn(self.feat_dim,
                                                             self.transformer_dim,
                                                             self.nneighbor,
                                                             expand_query_dim=False,
                                                             IFPS=self.init_block)

    def forward(self, pt_xyz, pt_feats, query_xyz, query_feat):
        query_feat, _ = self.query_self_attn(query_xyz, query_feat)  # self-attention
        query = torch.cat((query_xyz, query_feat), dim=-1)
        query_feat, _ = self.query_cross_attn(pt_xyz, pt_feats, query)  # cross-attention
        query_xyz = self.reg_branch(query_feat) + query_xyz

        return query_feat, query_xyz


class point_METRO_layer(nn.Module):

    def __init__(self, config):
        super(point_METRO_layer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward

        self.attn = BertAttention(config)
        self.cross_attn = BertAttention(config, position_embedding_type="absolute")
        self.vec_attn = pointer_layer(config)

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, query_feats, attention_mask, pt_feats, query_xyz, pt_xyz):
        self_attention_outputs = self.attn(
            hidden_states=query_feats,
            attention_mask=attention_mask,
            encoder_hidden_states=pt_feats,
            output_attentions=False,
        )

        attention_output = self_attention_outputs[0]

        # Feed the self_attention_output into cross_attention
        cross_attention_outputs = self.cross_attn(
            hidden_states=attention_output,
            attention_mask=attention_mask,
            encoder_hidden_states=pt_feats,
            output_attentions=False,
        )

        attention_output = cross_attention_outputs[0]  # This is the target output of the attention

        # ! Right here we embed the vec_attention part into the block
        # attention_output is the query_feats generated by the METRO block
        query_feats, query_xyz = self.vec_attn(pt_xyz, pt_feats, query_xyz, attention_output)

        # Only pass the query_feats into the Layer Norm and MLP part to update the query_feats
        query_feats = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, 1,
                                                query_feats)  # [B, 799, feat_dim]

        outputs = (query_feats, query_xyz)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class point_METRO_block(BertPreTrainedModel):
    """
        Modified the original METRO encoder to provide an interface for key value and query to be different.
    """

    def __init__(self, config):
        super(point_METRO_block, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = point_METRO_layer(config)
        self.pooler = BertPooler(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)  # Independent of K, V or Q
        self.input_dim = config.img_feature_dim
        self.parametric_output = config.parametric_output
        self.final_block = config.final_block

        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.embedding = nn.Linear(self.input_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.parametric_output:
            self.mano_layer = ManoLayer(joint_rot_mode="axisang",
                                        use_pca=False,
                                        mano_assets_root="assets/mano_v1_2",
                                        center_idx=config.center_idx,
                                        flat_hand_mean=True)
            self.flat_verts = nn.Linear(799, 1)
            self.mano_linear = nn.Linear(self.input_dim, 106)  # 106 = 16 * 6 + 10, 6D theta and beta

        # self.apply(self.init_weights)
        self.init_weights()

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_parametric_output(self, verts_feat, verts):
        verts_feat = verts_feat.reshape(-1, 799)
        flatten_feat = self.flat_verts(verts_feat)  # [799, B * FEAT_DIM] -> [1, B * FEAT_DIM]
        flatten_feat = flatten_feat.reshape(-1, self.input_dim)
        parametric_result = self.mano_linear(flatten_feat)  # [B, FEAT_DIM] -> [B, 106]
        pose_6d = parametric_result[:, :96]
        betas = parametric_result[:, 96:]
        pose_aa = rot6d_to_aa(pose_6d.view(-1, 16, 6)).view(-1, 48)
        mano_verts = self.mano_layer(pose_aa, betas).verts
        mano_joints = self.mano_layer(pose_aa, betas).joints
        verts[:, 21:, :] = mano_verts  # Update the verts from result of MANO layer
        verts[:, :21, :] = mano_joints
        return verts, pose_aa, betas

    def forward(self, query_xyz, query_feats, pt_xyz, pt_feats, head_mask=None):
        batch_size = len(query_feats)

        query_seq_length = len(query_feats[0])
        query_input_ids = torch.zeros([batch_size, query_seq_length], dtype=torch.long).to(query_feats.device)

        attention_mask = torch.ones_like(query_input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1))
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Project input token features to have specified hidden size
        query_embedding_output = self.embedding(query_feats)
        key_embedding_output = self.embedding(pt_feats)

        # ! We removed the positional embedding from this part.

        query_embeddings = self.dropout(query_embedding_output)
        key_embeddings = self.dropout(key_embedding_output)

        query_feats, query_xyz = self.encoder(query_feats=query_embeddings,
                                              attention_mask=extended_attention_mask,
                                              pt_feats=key_embeddings,
                                              query_xyz=query_xyz,
                                              pt_xyz=pt_xyz)

        if self.parametric_output and self.final_block:
            query_xyz, pred_pose, pred_shape = self.get_parametric_output(query_feats, query_xyz)
        else:
            pred_pose = None
            pred_shape = None

        return query_feats, query_xyz, pred_pose, pred_shape