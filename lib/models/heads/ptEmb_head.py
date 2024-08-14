import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from ...utils.config import CN
from ...utils.logger import logger
from ...utils.builder import HEAD
from ...utils.misc import param_size
from ...utils.transform import inverse_sigmoid, batch_cam_extr_transf, batch_cam_intr_projection
from ...utils.points_utils import sample_points_from_ball_query
from ..bricks.transformer import build_transformer
from ..layers.petr_transformer import SinePositionalEncoding3D
from pytorch3d.ops import ball_query
from ...utils.collation import generate_grid_sample_proj
from ...utils.transform import bchw_2_bhwc, denormalize
from manotorch.manolayer import ManoLayer
import imageio
import cv2
import time
import os


def viz_3d(image, joints_2d_proj):

    frame_1 = image.copy()
    for i in range(joints_2d_proj.shape[0]):
        cx = int(joints_2d_proj[i, 0])
        cy = int(joints_2d_proj[i, 1])
        cv2.circle(frame_1, (cx, cy), radius=1, thickness=-1, color=np.array([1.0, 0.0, 0.0]) * 255)

    img_list = [image, frame_1]
    comb_image = np.hstack(img_list)
    imageio.imwrite("tmp/img.png", comb_image)
    time.sleep(5)


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature**(2 * (torch.div(dim_t, 2, rounding_mode='floor')) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


class BasePointEmbedHead(nn.Module):

    def __init__(self, cfg: CN):
        super(BasePointEmbedHead, self).__init__()
        self.cfg_transformer = cfg.TRANSFORMER
        self.cfg_position_encoding = cfg.POSITIONAL_ENCODING

        self.with_position = cfg.WITH_POSITION
        self.with_multiview = cfg.WITH_MULTIVIEW
        self.num_query = cfg.NUM_QUERY  #  should equal to the n_vertices + n_joints,
        self.depth_num = cfg.DEPTH_NUM
        self.position_dim = 3 * self.depth_num
        self.position_range = cfg.POSITION_RANGE
        self.LID = cfg.LID
        self.depth_start = cfg.DEPTH_START  # The near - start depth
        self.depth_end = cfg.DEPTH_END  # The far - end depth
        self.embed_dims = cfg.EMBED_DIMS  # 256
        self.in_channels = cfg.IN_CHANNELS  # 128
        self.num_preds = cfg.NUM_PREDS

        # custom args
        self.center_shift = cfg.get("CENTER_SHIFT", False)

        self._build_head_module()
        self.init_weights()
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters, "
                    f"and got custom args: {self._str_custom_args()}")

    def _str_custom_args(self):
        return (f"center_shift: {self.center_shift}")

    def _build_head_module(self):
        self.center_shift_layer = nn.Sequential(nn.Linear(self.num_query, self.num_query), nn.ReLU(),
                                                nn.Linear(self.num_query, 1))

        self.positional_encoding = SinePositionalEncoding3D(num_feats=self.cfg_position_encoding.NUM_FEATS,
                                                            normalize=self.cfg_position_encoding.NORMALIZE)
        self.transformer = build_transformer(self.cfg_transformer)

        self.input_proj = nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
        self.reg_branches = nn.ModuleList()  # Defines the FFN
        for i in range(self.num_preds):
            reg_branch = nn.Sequential(nn.Linear(self.pt_feat_dim, self.pt_feat_dim), nn.ReLU(),
                                       nn.Linear(self.pt_feat_dim, 3))
            self.reg_branches.append(reg_branch)

        self.adapt_pos3d = nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims, kernel_size=1, stride=1, padding=0)
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 2, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.reference_embed = nn.Embedding(self.num_query, self.embed_dims)
        self.query_embedding = nn.Sequential(
            nn.Linear(6 + (self.embed_dims * 3 // 2), self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.pt_feat_dim),
        )

    def position_embeding(self, img_feat, img_metas, masks=None):
        eps = 1e-5
        inp_img_h, inp_img_w = img_metas['inp_img_shape']
        BN, C, H, W = img_feat.shape  # (BN, 128, 32, 32)
        B = len(img_metas["cam_view_num"])
        coords_h = torch.arange(H, device=img_feat.device).float() * inp_img_h / H  # U
        coords_w = torch.arange(W, device=img_feat.device).float() * inp_img_w / W  # V

        # Discretize the camera frustrum space
        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feat.device).float()
            index_plus1 = index + 1
            bin_size = (self.depth_end - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_plus1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feat.device).float()
            bin_size = (self.depth_end - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]

        # (W, H, D, 3)
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij')).permute(1, 2, 3, 0)

        # ===== 1. using camera intrinsic to convert UVD 2 XYZ  >>>>>>
        # This part first transforms the discretized camera frustrum points
        # from [U,V,D] -> [X,Y,Z]
        INTR = img_metas["cam_intr"]  # (BN, 3, 3)
        fx = INTR[..., 0, 0].unsqueeze(dim=-1)  # (BN, 1)
        fy = INTR[..., 1, 1].unsqueeze(dim=-1)  # (BN, 1)
        cx = INTR[..., 0, 2].unsqueeze(dim=-1)  # (BN, 1)
        cy = INTR[..., 1, 2].unsqueeze(dim=-1)  # (BN, 1)
        cam_param = torch.cat((fx, fy, cx, cy), dim=-1)  # (BN, 4)
        cam_param = cam_param.view(BN, 1, 1, 1, 4).repeat(1, W, H, D, 1)  # (BN, W, H, D, 4)

        coords_uv, coords_d = coords[..., :2], coords[..., 2:3]  # (W, H, D, 2), (W, H, D, 1)
        coords_uv = coords_uv.view(1, W, H, D, 2).repeat(BN, 1, 1, 1, 1)  # (BN, W, H, D, 2)
        coords_d = coords_d.view(1, W, H, D, 1).repeat(BN, 1, 1, 1, 1)  # (BN, W, H, D, 1)

        coords_uv = (coords_uv - cam_param[..., 2:4]) / cam_param[..., :2]  # (BN, W, H, D, 2)
        coords_xy = coords_uv * coords_d
        coords_z = coords_d
        coords = torch.cat((coords_xy, coords_z), dim=-1)  # (BN, W, H, D, 3)

        # ===== 2. using camera extrinsic to transfer childs' XYZ 2 parent's space >>>>>>
        # [X, Y, Z] in camera space -> [X, Y, Z] in world space
        EXTR = img_metas["cam_extr"]  # (BN, 4, 4)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords = coords.unsqueeze(-1)  # (BN, W, H, D, 4, 1)
        EXTR = EXTR.view(BN, 1, 1, 1, 4, 4).repeat(1, W, H, D, 1, 1)  # (BN, W, H, D, 4, 4)
        coords3d = torch.matmul(EXTR, coords).squeeze(-1)[..., :3]  # (BN, W, H, D, 3),  xyz in parent's space

        coords3d_absolute = copy.deepcopy(coords3d)
        # ===== 3. using position range to normalize coords3d
        #                 0     1     2     3     4     5
        # position_range [xmin, ymin, zmin, xmax, ymax, zmax]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / \
                                (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / \
                                (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / \
                                (self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)  # (BN, W, H, D, 3)

        coords3d_feat = coords3d.permute(0, 3, 4, 2, 1).contiguous().view(BN, -1, H, W)  # (BN, 3*D, H, W)
        coords3d_feat = inverse_sigmoid(coords3d_feat)

        coords_position_embeding = self.position_encoder(coords3d_feat)  # res: (BN, 256, H, W), 256 is self.embed_dims

        # ! coords_position_embedding represents f_M(M_{3d})
        # ! coords3d_absolute represents the camera frustrum points in shared world space
        return coords_position_embeding, coords3d, coords3d_absolute, coords_mask

    def init_weights(self):
        # The initialization for transformer is important, we leave it to the child class to decide.
        ### self.transformer.init_weights()
        nn.init.uniform_(self.reference_embed.weight.data, 0, 1)

    def forward(self, mlvl_feat, img_metas, reference_points, template_mesh, **kwargs):
        raise NotImplementedError(f"forward not implemented in base BasePointEmbedHead")


@HEAD.register_module()  # ptemb
class POEM_PositionEmbeddedAggregationHead(BasePointEmbedHead):

    def __init__(self, cfg: CN):
        self.nsample = cfg.N_SAMPLE
        self.radius = cfg.RADIUS_SAMPLE  # 0.2, set for ball query
        self.pt_feat_dim = cfg.POINTS_FEAT_DIM
        self.init_pt_feat_dim = cfg.INIT_POINTS_FEAT_DIM  # 8
        super(POEM_PositionEmbeddedAggregationHead, self).__init__(cfg)

    def _build_head_module(self):
        super(POEM_PositionEmbeddedAggregationHead, self)._build_head_module()
        self.transition_up = nn.Linear(self.init_pt_feat_dim, self.pt_feat_dim)
        logger.warning(f"{type(self).__name__} build done")

    def _str_custom_args(self):
        str_ = super(POEM_PositionEmbeddedAggregationHead, self)._str_custom_args()
        return str_ + ", " + f"init_pt_feat_dim: {self.init_pt_feat_dim}"

    def forward(self, mlvl_feat, img_metas, reference_points, template_mesh, **kwargs):
        """
            mlvl_feat: multi-view image features
            reference_points: reference mesh [X, V] where X -> V
            template_mesh: a flat MANO mesh
        """

        results = dict()
        x = mlvl_feat
        batch_size, num_cams = x.size(0), x.size(1)

        inp_img_w, inp_img_h = img_metas["inp_img_shape"]  #  (256, 256)
        ref_mesh_gt = img_metas["ref_mesh_gt"]
        inp_res = torch.Tensor([inp_img_w, inp_img_h]).to(x.device).float()
        masks = x.new_zeros((batch_size, num_cams, inp_img_h, inp_img_w))
        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])  # [B, N, 256, H, W]
        feat_dim = x.size(2)
        assert feat_dim == self.pt_feat_dim, \
            f"self.pt_feat_dim {self.pt_feat_dim} should be equal to feat_dim {feat_dim}"
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)

        # Here the positional embedding inherits from the base class

        coords_embed, coords3d, coords3d_abs, _ = self.position_embeding(mlvl_feat, img_metas, masks)

        sin_embed = self.positional_encoding(masks)

        # Applied another CNN layer
        sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())  # (B, N, 256, H, W)

        posi_embed = sin_embed + coords_embed
        x = x + posi_embed

        center_point = reference_points.mean(1).unsqueeze(1)  # [B, 1, 3]
        if self.center_shift == True:
            center_point = center_point + 0.01 * \
                           self.center_shift_layer(reference_points.transpose(1, 2)).transpose(1, 2)

        # x3d = self.input_proj_3Daware(x.flatten(0, 1))  # (BN, F*D, H, W)
        x3d = x.reshape(batch_size, num_cams, -1, self.depth_num, *x.shape[-2:])  # (B, N, F*D/D, D, H, W)
        x3d = x3d.permute(0, 1, 5, 4, 3, 2).contiguous()  # (B, N, W, H, D, F)
        assert x3d.shape[-1] == self.init_pt_feat_dim, \
            f"self.init_pt_feat_dim {self.init_pt_feat_dim} should be equal to x3d's last dim {x3d.shape[-1]}"

        reference_embed = self.reference_embed.weight  # (799, 3)
        reference_embed = pos2posemb3d(reference_embed)  # (799, 384)   384 <-- (self.embed_dims * 3 // 2)
        reference_embed = reference_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 799, 384)
        template_mesh = template_mesh.unsqueeze(0).expand(batch_size, -1, -1)  # template mesh  # (B, 799, 3)

        #*  normalize reference points into position range
        reference_points[..., 0:1] = (reference_points[..., 0:1] - self.position_range[0]) / \
                                        (self.position_range[3] - self.position_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] - self.position_range[1]) / \
                                        (self.position_range[4] - self.position_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] - self.position_range[2]) / \
                                        (self.position_range[5] - self.position_range[2])

        query_embeds = self.query_embedding(torch.cat([
            reference_embed,
            reference_points,
            template_mesh,
        ], dim=-1))  # (B, 799, embed_dims=32)

        # coords3d_abs: [B, N, 32, 32, 32, 3], --> [B, M, 3]
        coords3d_abs = coords3d_abs.reshape(batch_size, -1, 3)
        x3d = x3d.reshape(batch_size, -1, self.init_pt_feat_dim)
        randlist = torch.randperm(coords3d_abs.size(1))
        coords3d_abs = coords3d_abs[:, randlist, :]
        x3d = x3d[:, randlist, :]

        #  [B, nsample, 3],  [B, nsample, 32]
        pt_xyz, x3d_nsample = sample_points_from_ball_query(pt_xyz=coords3d_abs,
                                                            pt_feats=x3d,
                                                            center_point=center_point,
                                                            k=self.nsample,
                                                            radius=self.radius)

        pt_feats = self.transition_up(x3d_nsample)
        pt_xyz[..., 0:1] = (pt_xyz[..., 0:1] - self.position_range[0]) / \
                                        (self.position_range[3] - self.position_range[0])
        pt_xyz[..., 1:2] = (pt_xyz[..., 1:2] - self.position_range[1]) / \
                                        (self.position_range[4] - self.position_range[1])
        pt_xyz[..., 2:3] = (pt_xyz[..., 2:3] - self.position_range[2]) / \
                                        (self.position_range[5] - self.position_range[2])

        interm_ref_pts = self.transformer(
            pt_xyz=pt_xyz,  # [B, 2048, 3]
            pt_feats=pt_feats,  # [B, 2048, 256]
            query_emb=query_embeds,  # [B, 799, 256]
            query_xyz=reference_points,  # [B, 799, 3]
            reg_branches=self.reg_branches,
        )

        interm_ref_pts = torch.nan_to_num(interm_ref_pts)
        all_coords_preds = interm_ref_pts  # (N_Decoder, BS, NUM_QUERY, 3)

        # scale all the joitnts and vertices to the camera space
        # position_range [xmin, ymin, zmin, xmax, ymax, zmax]
        all_coords_preds[..., 0:1] = \
            all_coords_preds[..., 0:1] * (self.position_range[3] - self.position_range[0]) + self.position_range[0]
        all_coords_preds[..., 1:2] = \
            all_coords_preds[..., 1:2] * (self.position_range[4] - self.position_range[1]) + self.position_range[1]
        all_coords_preds[..., 2:3] = \
            all_coords_preds[..., 2:3] * (self.position_range[5] - self.position_range[2]) + self.position_range[2]

        results["all_coords_preds"] = all_coords_preds
        return results


@HEAD.register_module()  # proj_selfagg
class POEM_Projective_SelfAggregation_Head(BasePointEmbedHead):

    def __init__(self, cfg: CN):
        self.nsample = cfg.N_SAMPLE
        self.radius = cfg.RADIUS_SAMPLE
        self.pt_feat_dim = cfg.POINTS_FEAT_DIM  # 256
        self.merge_mode = cfg.get("CAM_FEAT_MERGE", "sum")  # By default, attention
        self.query_type = cfg.get("QUERY_TYPE", "KPT")
        # POEM: I cat V_tmpl cat V_init
        # KPT: I
        # MVP: g + I
        # METRO: g cat V_tmpl
        super(POEM_Projective_SelfAggregation_Head, self).__init__(cfg)

    def _build_head_module(self):
        super(POEM_Projective_SelfAggregation_Head, self)._build_head_module()

        # Network G
        self.merge_net_feature = nn.ModuleList()
        self.merge_net_feature.append(
            nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(),
                          nn.Linear(self.embed_dims, self.embed_dims // 2)))
        self.merge_net_feature.append(
            nn.Sequential(nn.Linear(self.embed_dims // 2, self.embed_dims // 2), nn.ReLU(),
                          nn.Linear(self.embed_dims // 2, self.embed_dims)))
        self.merge_net_query_feature = nn.ModuleList()
        self.merge_net_query_feature.append(
            nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(),
                          nn.Linear(self.embed_dims, self.embed_dims // 2)))
        self.merge_net_query_feature.append(
            nn.Sequential(nn.Linear(self.embed_dims // 2, self.embed_dims // 2), nn.ReLU(),
                          nn.Linear(self.embed_dims // 2, self.embed_dims)))

        self.layer_global_feat = nn.Linear(512, self.embed_dims)
        if self.query_type == "POEM":
            self.query_embedding = nn.Sequential(
                nn.Linear(6 + self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.pt_feat_dim),
            )
        elif self.query_type == "KPT":
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.pt_feat_dim),
            )
        elif self.query_type == "MVP":
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.pt_feat_dim),
            )
        elif self.query_type == "METRO":
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims + 3, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.pt_feat_dim),
            )
        else:
            raise ValueError(f"no such query_type: {self.query_type}")
        logger.info(f"{type(self).__name__} got query_type: {self.query_type}")
        logger.warning(f"{type(self).__name__} build done")

    def _str_custom_args(self):
        str_ = super()._str_custom_args()
        return str_ + ", " + f"agg_merge_mode: {self.merge_mode}"

    def merge_features(self, q, merge_net, master_id):
        """
            q: [B, nsample, N, 256]
            q_merged: [B, nsample, 256]
        """
        master_is_zero = torch.sum(master_id)
        assert master_is_zero == 0, "only support master_id is 0"
        q1 = q[:, :, 0, :]
        q = merge_net[0](q)
        master_features = q[:, :, 0, :]  # [B, nsample, 128]
        other_features = q[:, :, 1:, :]  # [B, nsample, 7, 128]
        q = torch.matmul(other_features, master_features.unsqueeze(-1))  # [B, nsample, 7, 1]
        q = torch.matmul(other_features.transpose(2, 3), q).squeeze(-1)
        q_merged = merge_net[1](q)  # [B, nsample, 256]
        q_merged = q1 + q_merged
        return q_merged

    def sample_points_from_ball_query(self, pt_xyz, center_point, k, radius):
        _, ball_idx, xyz = ball_query(center_point, pt_xyz, K=k, radius=radius, return_nn=True)
        invalid = torch.sum(ball_idx == -1) > 0
        if invalid:
            # NOTE: sanity check based on bugs reported Oct. 24
            logger.warning(f"ball query returns {torch.sum(ball_idx == -1)} / {torch.numel(ball_idx)} -1 in its index, "
                           f"which means you need to increase raidus or decrease K")
        xyz = xyz.squeeze(1)
        return xyz

    def generate_query(self, reference_embed, reference_points, template_mesh, global_feat):
        if self.query_type == "POEM":
            query_embeds = self.query_embedding(torch.cat([
                reference_embed,
                reference_points,
                template_mesh,
            ], dim=-1))  # (B, 799, pt_feat_dim)
        elif self.query_type == "KPT":  # keypoint query mode (in the paper, S^J)
            query_embeds = self.query_embedding(reference_embed)  # (B, 799, pt_feat_dim)
        elif self.query_type == "MVP":
            query_embeds = self.query_embedding(global_feat + reference_embed)
        elif self.query_type == "METRO":
            query_embeds = self.query_embedding(torch.cat([
                global_feat,
                template_mesh,
            ], dim=-1))  # (B, 799, pt_feat_dim)
        else:
            raise ValueError(f"no such query_type: {self.query_type}")
        return query_embeds

    def forward(self, mlvl_feat, img_metas, reference_points, template_mesh, **kwargs):
        results = dict()
        x = mlvl_feat  # mlvl_feat := (BN, 128, 32, 32) / prev (B, N, 128, 32, 32)
        BN = x.shape[0]
        batch_size = len(img_metas["master_id"])

        global_feat = kwargs.get("global_feat", None)  # (BN, 512)
        if global_feat != None and self.query_type != "KPT":
            global_feat = self.layer_global_feat(global_feat)  # (BN, 256)
            assert global_feat.shape[0] % batch_size == 0
            global_feat = global_feat.reshape(batch_size, -1, global_feat.shape[1])  # (B, N, 256)
            global_feat = global_feat.sum(1)  # (B, 256) -> (B, 799, 256)
            global_feat = global_feat.unsqueeze(1).repeat(1, self.num_query, 1)

        inp_img_w, inp_img_h = img_metas["inp_img_shape"]  #  (256, 256)
        inp_res = torch.Tensor([inp_img_w, inp_img_h]).to(x.device).float()
        masks = x.new_zeros((BN, inp_img_h, inp_img_w))
        x = self.input_proj(x)  # No need for flatten as x is of shape [BN, 128, 32, 32]

        x = x.view(-1, *x.shape[-3:])  # [BN, 256, H, W]
        feat_dim = x.size(1)
        assert feat_dim == self.pt_feat_dim, "self.pt_feat_dim should be equal to feat_dim"

        # interpolate masks to have the same spatial shape with x, require input of [B, N, img_h, img_W]. Iterate over each subbatch
        masks_batch = []
        for i in range(batch_size):
            start_idx = np.sum(img_metas["cam_view_num"][:i])
            end_idx = np.sum(img_metas["cam_view_num"][:i + 1])
            masks_sub = masks[start_idx:end_idx].unsqueeze(0)
            masks_sub = F.interpolate(masks_sub, size=x.shape[-2:]).to(torch.bool)
            masks_batch.append(masks_sub)
        masks = torch.cat(masks_batch, dim=1).squeeze(0)  # Shape [14, 32, 32]

        # coords_embed is the result for f_m(M_{3d}) := [BN, 256, 32, 32]
        # coords3d_abs is the coordinates M_{3d} in shared world space := [BN, W, H, D, 3]
        coords_embed, coords3d, coords3d_abs, _ = self.position_embeding(mlvl_feat, img_metas, masks)

        # Iterate over masks for positional_coding. PETR_Transformer require masks
        # of shape (B, N, H, W). We also pass the adapt_pos3d layer here before concating later
        sin_embed = []
        for i in range(batch_size):
            start_idx = np.sum(img_metas["cam_view_num"][:i])
            end_idx = np.sum(img_metas["cam_view_num"][:i + 1])
            sin_embed_sub = self.positional_encoding(masks[start_idx:end_idx].unsqueeze(0)).squeeze(0)
            sin_embed_sub = self.adapt_pos3d(sin_embed_sub)
            sin_embed.append(sin_embed_sub)
        sin_embed = torch.concat(sin_embed, dim=0)  # (BN, 256, H, W)

        # posi_embed consists of sin and MLP functions
        posi_embed = sin_embed + coords_embed
        x = x + posi_embed

        center_point = reference_points.mean(1).unsqueeze(1)  # [B, 1, 3]
        if self.center_shift == True:
            center_point = center_point + 0.01 * \
                           self.center_shift_layer(reference_points.transpose(1, 2)).transpose(1, 2)

        # reference_points := [B, 799, 3]
        reference_embed = self.reference_embed.weight  # (799, 256)
        reference_embed = reference_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 799, 256)
        template_mesh = template_mesh.unsqueeze(0).expand(batch_size, -1, -1)  # template mesh  # (B, 799, 3)

        #*  normalize reference points into position range
        reference_points[..., 0:1] = (reference_points[..., 0:1] - self.position_range[0]) / \
                                        (self.position_range[3] - self.position_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] - self.position_range[1]) / \
                                        (self.position_range[4] - self.position_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] - self.position_range[2]) / \
                                        (self.position_range[5] - self.position_range[2])

        query_embeds = self.generate_query(
            reference_embed=reference_embed,
            reference_points=reference_points,
            template_mesh=template_mesh,
            global_feat=global_feat,
        )  # (B, 799, pt_feat_dim)

        # would require size % batch == 0 here, so apply iteration.
        coords3d_nsample = []
        for i in range(batch_size):
            start_idx = np.sum(img_metas["cam_view_num"][:i])
            end_idx = np.sum(img_metas["cam_view_num"][:i + 1])
            coords3d_abs_sub = coords3d_abs[start_idx:end_idx].reshape(1, -1, 3)  # prev. (B, -1, 3)

            randlist = torch.randperm(coords3d_abs_sub.size(1))
            coords3d_abs_sub = coords3d_abs_sub[:, randlist, :]
            coords3d_nsample_sub = self.sample_points_from_ball_query(coords3d_abs_sub, center_point[i].unsqueeze(0),
                                                                      self.nsample, self.radius)
            coords3d_nsample.append(coords3d_nsample_sub)

        coords3d_nsample = torch.concat(coords3d_nsample, dim=0)  # (B, nsample, 3)
        pt_xyz = coords3d_nsample  # (B, nsample, 3)

        # coords3d_nsample -> [B, N, nsample, 3], coords3d_nsample_project: [B, N, nsample, 2]
        # Here, the transformation would require iteration again.
        coords3d_nsample_project = []
        for i in range(batch_size):
            start_idx = np.sum(img_metas["cam_view_num"][:i])
            end_idx = np.sum(img_metas["cam_view_num"][:i + 1])

            # [B, n, 3] -> [nsample, 3] -> [N, nsample, 3]
            # -> [1, N, nsample, 3] / prev. (B, N, nsample, 3)
            coords3d_nsample_sub = coords3d_nsample[i].repeat(img_metas["cam_view_num"][i], 1, 1).unsqueeze(0)
            cam_extr_sub = img_metas["cam_extr"][start_idx:end_idx].unsqueeze(0)
            cam_intr_sub = img_metas["cam_intr"][start_idx:end_idx].unsqueeze(0)

            coords3d_nsample_project_sub = batch_cam_extr_transf(torch.linalg.inv(cam_extr_sub), coords3d_nsample_sub)
            coords3d_nsample_project_sub = batch_cam_intr_projection(cam_intr_sub, coords3d_nsample_project_sub)
            coords3d_nsample_project.append(coords3d_nsample_project_sub)

        coords3d_nsample_project = torch.concat(coords3d_nsample_project, dim=1).squeeze(0)
        coords3d_nsample_project = coords3d_nsample_project.unsqueeze(-2)  # WH —> uv

        coords3d_nsample_project = torch.einsum("bijk, k->bijk", coords3d_nsample_project,
                                                1.0 / inp_res)  # TENSOR (B, N,  nsample, 2), [0 ~ 1]
        coords3d_nsample_project = coords3d_nsample_project * 2 - 1
        coords3d_project_invalid = (coords3d_nsample_project > 1.0) | (coords3d_nsample_project < -1.0)

        # NOTE: sanity check based on bugs reported Oct. 24
        ratio_invalid = torch.sum(coords3d_project_invalid) / torch.numel(coords3d_project_invalid)
        if ratio_invalid > 0.3 and ratio_invalid <= 0.5:
            logger.warning(f"Projection returns {torch.sum(coords3d_project_invalid)} / "
                           f"{torch.numel(coords3d_project_invalid)} outsiders in its resutls, "
                           f"may be a bug !")
        if ratio_invalid > 0.5:
            raise ValueError(f"Too many invalid projective points, "
                             f"consider reduce the RADIUS_SAMPLE : {self.radius}, "
                             f"and check center_points's validness !")

        # pt_sampled_feats: [B, nsample, N, 256], sin_embed_sample: [B, nsample, N, 256]

        # operations of the ref_proj_2d require iteration
        ref_proj_2d = []
        for i in range(batch_size):
            start_idx = np.sum(img_metas["cam_view_num"][:i])
            end_idx = np.sum(img_metas["cam_view_num"][:i + 1])

            # [B, n, 3] -> [nsample, 3] -> [N, nsample, 3]
            # -> [1, N, nsample, 3] / prev. (B, N, nsample, 3)
            ref_proj_2d_sub = reference_points[i].repeat(img_metas["cam_view_num"][i], 1, 1).unsqueeze(0)
            cam_extr_sub = img_metas["cam_extr"][start_idx:end_idx].unsqueeze(0)
            cam_intr_sub = img_metas["cam_intr"][start_idx:end_idx].unsqueeze(0)

            ref_proj_2d_sub = batch_cam_extr_transf(torch.linalg.inv(cam_extr_sub), ref_proj_2d_sub)
            ref_proj_2d_sub = batch_cam_intr_projection(cam_intr_sub, ref_proj_2d_sub)
            ref_proj_2d.append(ref_proj_2d_sub)

        ref_proj_2d = torch.concat(ref_proj_2d, dim=1).squeeze(0)
        ref_proj_2d = ref_proj_2d.unsqueeze(-2)  # WH —> uv
        ref_proj_2d = torch.einsum("bijk, k->bijk", ref_proj_2d, 1.0 / inp_res)  # TENSOR (B, N,  nsample, 2), [0 ~ 1]
        ref_proj_2d = ref_proj_2d * 2 - 1

        query_feat = F.grid_sample(x, ref_proj_2d, align_corners=False)\
                    .squeeze(-1).reshape(BN, feat_dim, self.num_query)
        pt_sampled_feats = F.grid_sample(x, coords3d_nsample_project, align_corners=False)\
                            .squeeze(-1).reshape(BN, feat_dim, self.nsample)
        pt_sampled_emb = F.grid_sample(posi_embed, coords3d_nsample_project, align_corners=False)\
                            .squeeze(-1).reshape(BN, feat_dim, self.nsample)

        if self.merge_mode == "attn":
            master_id = img_metas["master_id"]
            # pt_sampled_feats := [BN, feat_dim, nsample]
            # query_feat := [BN, feat_dim, nsample]

            pt_sampled_feats_full = []
            query_feats_full = []
            for i in range(batch_size):
                start_idx = np.sum(img_metas["cam_view_num"][:i])
                end_idx = np.sum(img_metas["cam_view_num"][:i + 1])

                pt_sampled_feats_sub = pt_sampled_feats[start_idx:end_idx].view(1, -1, img_metas["cam_view_num"][i],
                                                                                pt_sampled_feats.size(1))

                pt_sampled_feats_sub = self.merge_features(pt_sampled_feats_sub, self.merge_net_feature,
                                                           torch.Tensor([master_id[i]]))

                query_feat_sub = query_feat[start_idx:end_idx].view(1, -1, img_metas["cam_view_num"][i],
                                                                    pt_sampled_feats.size(1))
                query_feat_sub = self.merge_features(query_feat_sub, self.merge_net_feature,
                                                     torch.Tensor([master_id[i]]))

                query_feats_full.append(query_feat_sub)
                pt_sampled_feats_full.append(pt_sampled_feats_sub)

            pt_sampled_feats = torch.concat(pt_sampled_feats_full)
            query_feat = torch.concat(query_feats_full)
        elif self.merge_mode == "sum":
            # ! Only available for constant num of cameras
            pt_sampled_feats = torch.sum(pt_sampled_feats, dim=-2)  # [B, nsample, 256]
            query_feat = torch.sum(query_feat, dim=-2)
        else:
            raise ValueError(f"CAM_FEAT_MERGE must in [attn, sum], default sum, got {self.merge_mode}")

        # pt_sampled_feats := [B, nsample, 256]
        # query_feat := [B, nsample, 256]

        pt_xyz[..., 0:1] = (pt_xyz[..., 0:1] - self.position_range[0]) / \
                                        (self.position_range[3] - self.position_range[0])
        pt_xyz[..., 1:2] = (pt_xyz[..., 1:2] - self.position_range[1]) / \
                                        (self.position_range[4] - self.position_range[1])
        pt_xyz[..., 2:3] = (pt_xyz[..., 2:3] - self.position_range[2]) / \
                                        (self.position_range[5] - self.position_range[2])

        # pt_sampled_emb requires reshape into [B, nsample, feat_size]
        # Therefore, we have to do a final iteration
        pt_embed = []
        for i in range(batch_size):
            start_idx = np.sum(img_metas["cam_view_num"][:i])
            end_idx = np.sum(img_metas["cam_view_num"][:i + 1])
            pt_sampled_emb_sub = pt_sampled_emb[start_idx:end_idx].unsqueeze(0)
            pt_sampled_emb_sub = pt_sampled_emb_sub.sum(1).view(1, -1, feat_dim)
            pt_embed.append(pt_sampled_emb_sub)
        pt_embed = torch.concat(pt_embed, dim=0)

        # Now the transformer can be trained in batches.
        interm_ref_pts = self.transformer(
            pt_xyz=pt_xyz,  # [B, feat_dim, 3]
            pt_feats=pt_sampled_feats,  # [B, feat_dim, 256]
            pt_embed=pt_sampled_emb_sub,  # [B, feat_dim, 256]
            query_feat=query_feat,
            query_emb=query_embeds,  # [B, 799, 256]
            query_xyz=reference_points,  # [B, 799, 3]
            reg_branches=self.reg_branches,
        )

        interm_ref_pts = torch.nan_to_num(interm_ref_pts)
        all_coords_preds = interm_ref_pts  # (N_Decoder, BS, NUM_QUERY, 3)

        # scale all the joitnts and vertices to the camera space
        # position_range [xmin, ymin, zmin, xmax, ymax, zmax]
        all_coords_preds[..., 0:1] = \
            all_coords_preds[..., 0:1] * (self.position_range[3] - self.position_range[0]) + self.position_range[0]
        all_coords_preds[..., 1:2] = \
            all_coords_preds[..., 1:2] * (self.position_range[4] - self.position_range[1]) + self.position_range[1]
        all_coords_preds[..., 2:3] = \
            all_coords_preds[..., 2:3] * (self.position_range[5] - self.position_range[2]) + self.position_range[2]

        results["all_coords_preds"] = all_coords_preds
        return results


@HEAD.register_module()
class POEM_Generalized_Head(BasePointEmbedHead):

    def __init__(self, cfg: CN):
        self.nsample = cfg.N_SAMPLE
        self.radius = cfg.RADIUS_SAMPLE
        self.pt_feat_dim = cfg.POINTS_FEAT_DIM  # 256
        self.merge_mode = cfg.get("CAM_FEAT_MERGE", "attn")  # By default, attention
        self.query_type = cfg.get("QUERY_TYPE", "POEM")
        self.PETR_embedding = cfg.get("PETR_EMBEDDING", False)
        self.parametric_output = cfg.TRANSFORMER.get("PARAMETRIC_OUTPUT", False)
        self.transformer_center_idx = cfg.TRANSFORMER.get("TRANSFORMER_CENTER_IDX", 9)
        super(POEM_Generalized_Head, self).__init__(cfg)

    def _build_head_module(self):
        super(POEM_Generalized_Head, self)._build_head_module()

        # Network G
        self.merge_net_feature = nn.ModuleList()
        self.merge_net_feature.append(
            nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(),
                          nn.Linear(self.embed_dims, self.embed_dims // 2)))
        self.merge_net_feature.append(
            nn.Sequential(nn.Linear(self.embed_dims // 2, self.embed_dims // 2), nn.ReLU(),
                          nn.Linear(self.embed_dims // 2, self.embed_dims)))
        self.merge_net_query_feature = nn.ModuleList()

        # Configuration for the merge net
        self.merge_net_query_feature = nn.ModuleList()
        self.merge_net_query_feature.append(
            nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(),
                          nn.Linear(self.embed_dims, self.embed_dims // 2)))
        self.merge_net_query_feature.append(
            nn.Sequential(nn.Linear(self.embed_dims // 2, self.embed_dims // 2), nn.ReLU(),
                          nn.Linear(self.embed_dims // 2, self.embed_dims)))

        self.layer_global_feat = nn.Linear(512, self.embed_dims)

        assert self.query_type == "KPT"
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.pt_feat_dim),
        )

        # Configuration for the query_feat_embedding
        self.query_feat_embedding = nn.Embedding(799, self.pt_feat_dim)  # (799, 256)

        # A mano layer for creating template mesh
        self.mano_layer = ManoLayer(joint_rot_mode="axisang",
                                    use_pca=False,
                                    mano_assets_root="assets/mano_v1_2",
                                    center_idx=self.transformer_center_idx,
                                    flat_hand_mean=True)

        logger.info(f"{type(self).__name__} got query_type: {self.query_type}")
        logger.warning(f"{type(self).__name__} build done")

    def _str_custom_args(self):
        str_ = super()._str_custom_args()
        return str_ + ", " + f"agg_merge_mode: {self.merge_mode}"

    def merge_features_mv(self, q, merge_net, master_id):
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

    def merge_features_sv(self, q, merge_net, master_id):
        master_is_zero = torch.sum(master_id)
        assert master_is_zero == 0, "only support master_id is 0"

        master_features = merge_net[0](q)
        q_output = merge_net[1](master_features)  # [B, nsample, 256]
        q_output = q + q_output
        return q_output

    def _generate_random_basis(self, n_points, n_dims, radius, device):
        """
        Sample uniformly from d-dimensional unit ball
        Adapted from https://github.com/sergeyprokudin/bps/blob/master/bps/bps.py
        """
        # sample point from d-sphere
        x = torch.randn(n_points, n_dims).to(device)
        x_norms = torch.norm(x, dim=1).reshape((-1, 1))
        x_unit = x / x_norms

        # now sample radii uniformly
        r = torch.rand(n_points, 1).to(device)
        u = torch.pow(r, 1.0 / n_dims)
        x = radius * x_unit * u

        return x

    def get_bps(self, hand_root, nsample, radius, device):
        bps_dir = os.path.join("assets", "bps.npy")
        batch_size = hand_root.shape[0]
        if not os.path.exists(bps_dir):
            # No bps points have been generated
            bps_points_root = self._generate_random_basis(nsample, 3, radius, device)  # centered in (0,0) and radius
            bps_points_root = bps_points_root.unsqueeze(0)  # (1, N, 3)
            bps_points_dump = bps_points_root.cpu().detach().numpy()
            np.save(bps_dir, bps_points_dump)  # dump BPS points of shape (1, N, 3)

            # repeat bps_points by batch_size on dim 0
            bps_points_root = bps_points_root.repeat(batch_size, 1, 1)
            bps_points_world = bps_points_root + hand_root.unsqueeze(1).repeat(1, nsample, 1)
        else:
            # load the bps points (should remain the same each time)
            bps_points_load = np.load(bps_dir)  # (1, N, 3)
            bps_points_root = torch.Tensor(bps_points_load).to(device)  # root-relative coords
            bps_points_root = bps_points_root.repeat(batch_size, 1, 1)  # (B, N, 3)
            bps_points_world = bps_points_root + hand_root.unsqueeze(1).repeat(1, nsample, 1)  # world coords
        return bps_points_world

    def generate_query(self, reference_embed):
        # Use KPT as default
        query_embeds = self.query_embedding(reference_embed)
        return query_embeds

    def _debug_viz(self, debug_metas, src_points_sub, idx=0):
        # Visualize the first view
        img_single = debug_metas["img"][idx]
        image = img_single.detach().cpu().unsqueeze(0)
        image = bchw_2_bhwc(denormalize(image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
        image = image.mul_(255.0).view(1, 256, 256, 3).squeeze(0).numpy().astype(np.uint8)  # (B, H, W, 3)
        vis_points = src_points_sub[idx]
        viz_3d(image, vis_points)

    def forward(self, mlvl_feat, img_metas, reference_joints, **kwargs):
        results = dict()
        x = mlvl_feat  # mlvl_feat := (BN, 128, 32, 32) / prev (B, N, 128, 32, 32)
        BN = x.shape[0]
        batch_size = len(img_metas["master_id"])

        inp_img_w, inp_img_h = img_metas["inp_img_shape"]  #  (256, 256)
        inp_res = torch.Tensor([inp_img_w, inp_img_h]).to(x.device).float()
        img_metas["inp_res"] = inp_res
        masks = x.new_zeros((BN, inp_img_h, inp_img_w))
        x = self.input_proj(x)  # No need for flatten as x is of shape [BN, 128, 32, 32]

        x = x.view(-1, *x.shape[-3:])  # [BN, 256, H, W]
        feat_dim = x.size(1)
        assert feat_dim == self.pt_feat_dim, "self.pt_feat_dim should be equal to feat_dim"

        # interpolate masks to have the same spatial shape with x, require input of [B, N, img_h, img_W]. Iterate over each subbatch
        masks_batch = []
        for i in range(batch_size):
            start_idx = np.sum(img_metas["cam_view_num"][:i])
            end_idx = np.sum(img_metas["cam_view_num"][:i + 1])
            masks_sub = masks[start_idx:end_idx].unsqueeze(0)
            masks_sub = F.interpolate(masks_sub, size=x.shape[-2:]).to(torch.bool)
            masks_batch.append(masks_sub)
        masks = torch.cat(masks_batch, dim=1).squeeze(0)  # Shape [14, 32, 32]

        # Iterate over masks for positional_coding. PETR_Transformer require masks
        # of shape (B, N, H, W). We also pass the adapt_pos3d layer here before concating later
        sin_embed = []
        for i in range(batch_size):
            start_idx = np.sum(img_metas["cam_view_num"][:i])
            end_idx = np.sum(img_metas["cam_view_num"][:i + 1])
            sin_embed_sub = self.positional_encoding(masks[start_idx:end_idx].unsqueeze(0)).squeeze(0)
            sin_embed_sub = self.adapt_pos3d(sin_embed_sub)
            sin_embed.append(sin_embed_sub)
        sin_embed = torch.concat(sin_embed, dim=0)  # (BN, 256, H, W)

        # posi_embed consists of sin functions only by default
        posi_embed = sin_embed

        if self.PETR_embedding:
            coords_embed, coords3d, coords3d_abs, _ = self.position_embeding(mlvl_feat, img_metas, masks)
            posi_embed += coords_embed

        # Once x, i.e. F_{3d} is derived, we don't need discretized camera frustum now
        x = x + posi_embed

        # >>>>>>>>   Generate the BPS set from the reference_joint in shared world space
        reference_hand_center = reference_joints[:, 9, :]
        bps_points = self.get_bps(reference_hand_center, self.nsample, self.radius, x.device)  # (B, nsample, 3)
        pt_xyz = bps_points  # Reserved before the projection, required for Point Transformer

        # Project bps_points with regard to different cameras -> uvd
        bps_points_project = generate_grid_sample_proj(bps_points, img_metas)

        bps_points_project = bps_points_project.unsqueeze(-2)  # used for grid_sampling
        bps_points_project = torch.einsum("bijk, k->bijk", bps_points_project,
                                          1.0 / inp_res)  # TENSOR (BN,  nsample, 2), [0 ~ 1]
        bps_points_project = bps_points_project * 2 - 1

        # Create a template mesh for initializing the points
        template_pose = torch.zeros((1, 48)).to(x.device)
        template_betas = torch.zeros((1, 10)).to(x.device)
        mano_out = self.mano_layer(template_pose, template_betas)
        template_vertices = mano_out.verts
        template_3d_joints = mano_out.joints
        template_mesh = torch.cat([template_3d_joints, template_vertices], dim=1).repeat(batch_size, 1,
                                                                                         1)  # (B, 799, 3)
        reference_points = reference_hand_center.unsqueeze(1).repeat(
            1, 799, 1) + template_mesh  # hand root centered at ref_hand_center

        # reference_points = torch.concat([reference_joints, reference_verts], dim=1)
        # >>>>>>>>> End section

        # >>>>>>>>>     Merging multi-view features of BPS points
        bps_sampled_feat = F.grid_sample(x, bps_points_project, align_corners=False)\
                            .squeeze(-1).reshape(BN, feat_dim, self.nsample)

        assert self.merge_mode == "attn"
        master_id = img_metas["master_id"]
        # pt_sampled_feats := [BN, feat_dim, nsample]
        # query_feat := [BN, feat_dim, nsample]

        bps_sampled_feats_full = []

        for i in range(batch_size):
            start_idx = np.sum(img_metas["cam_view_num"][:i])
            end_idx = np.sum(img_metas["cam_view_num"][:i + 1])

            bps_sampled_feats_sub = bps_sampled_feat[start_idx:end_idx].view(1, -1, img_metas["cam_view_num"][i],
                                                                             bps_sampled_feat.size(1))
            if end_idx - start_idx == 1:
                bps_sampled_feats_sub = bps_sampled_feats_sub.squeeze(2)
                bps_sampled_feats_sub = self.merge_features_sv(bps_sampled_feats_sub, self.merge_net_feature,
                                                               torch.Tensor([master_id[i]]))
            else:
                bps_sampled_feats_sub = self.merge_features_mv(bps_sampled_feats_sub, self.merge_net_feature,
                                                               torch.Tensor([master_id[i]]))

            bps_sampled_feats_full.append(bps_sampled_feats_sub)

        bps_sampled_feat = torch.concat(bps_sampled_feats_full)
        # >>>>>>>>>>>>>>>  End Section

        # The features for query points are initialized as nn.Embedding.
        query_feat_embedding = self.query_feat_embedding.weight  # (799, 256)
        query_feat = query_feat_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 799, 256)

        # normalize pt_xyz to [-1, 1] and bps_xyz to [-1, 1]
        pt_xyz = (pt_xyz - reference_hand_center.unsqueeze(1).repeat(1, self.nsample, 1)) / self.radius
        reference_points = (reference_points - reference_hand_center.unsqueeze(1).repeat(1, 799, 1)) / self.radius

        # Now the transformer can be trained in batches.
        interm_ref_pts, pred_pose, pred_shape = self.transformer(
            pt_xyz=pt_xyz,  # [B, nsample, 3]
            pt_feats=bps_sampled_feat,  # [B, nsample, 256]
            query_feat=query_feat,
            query_xyz=reference_points)

        interm_ref_pts = torch.nan_to_num(interm_ref_pts)
        all_coords_preds = interm_ref_pts  # (NUM_TRANSFORMER_BLOCKS, BS, NUM_QUERY, 3)

        # scale all the joints and vertices to the camera space for non-parametric-case and the preds back and add the offsetm
        reference_hand_center = reference_hand_center.unsqueeze(1).unsqueeze(0)  # (1, B, 1, 3)\
        if not self.parametric_output:
            all_coords_preds = all_coords_preds * self.radius + reference_hand_center.repeat(
                all_coords_preds.shape[0], 1, all_coords_preds.shape[2], 1)
        else:
            # Only the last layer returns the unscaled result, so previous layers still need to be scaled back
            all_coords_preds[:
                             -1, :, :, :] = all_coords_preds[:-1, :, :, :] * self.radius + reference_hand_center.repeat(
                                 all_coords_preds.shape[0] - 1, 1, all_coords_preds.shape[2], 1)
            # We won't make the scale for the last layer as it returns the result from mano layer
            all_coords_preds[-1, :, :, :] = all_coords_preds[-1, :, :, :] + reference_hand_center.repeat(1, 1, 799, 1)

        results["all_coords_preds"] = all_coords_preds
        if self.parametric_output:
            results["pred_pose"] = pred_pose.reshape(-1, 16, 3)
            results["pred_shape"] = pred_shape.reshape(-1, 10)
        return results
