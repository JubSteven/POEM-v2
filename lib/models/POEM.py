import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from manotorch.manolayer import ManoLayer

from termcolor import cprint
from torch import distributed as dist
from ..utils.triangulation import batch_triangulate_dlt_torch
from ..metrics.basic_metric import LossMetric
from ..metrics.mean_epe import MeanEPE
from ..metrics.pa_eval import PAEval
from ..utils.builder import MODEL
from ..utils.logger import logger
from ..utils.misc import param_size
from ..utils.net_utils import load_weights
from ..utils.recorder import Recorder
from ..utils.transform import (batch_cam_extr_transf, batch_cam_intr_projection, rot6d_to_aa, mano_to_openpose,
                               batch_persp_project)
from ..viztools.draw import draw_batch_joint_images, draw_batch_verts_images
from .backbones import build_backbone
from .bricks.conv import ConvBlock
from .heads import build_head
from .integal_pose import integral_heatmap2d
from .model_abc import ModelABC


@MODEL.register_module()
class PtEmbedMultiviewStereoV2(ModelABC):

    def __init__(self, cfg):
        super(PtEmbedMultiviewStereoV2, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.train_cfg = cfg.TRAIN
        self.data_preset_cfg = cfg.DATA_PRESET
        self.num_joints = cfg.DATA_PRESET.NUM_JOINTS
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        self.joints_loss_type = cfg.LOSS.get("JOINTS_LOSS_TYPE", "l2")
        self.verts_loss_type = cfg.LOSS.get("VERTICES_LOSS_TYPE", "l1")
        self.pred_joints_from_mesh = cfg.get("PRED_JOINTS_FROM_MESH", True)
        self.parametric_output = cfg.HEAD.TRANSFORMER.get("PARAMETRIC_OUTPUT", False)
        self.transformer_center_idx = cfg.HEAD.TRANSFORMER.get("TRANSFORMER_CENTER_IDX", 9)

        self.img_backbone = build_backbone(cfg.BACKBONE, data_preset=self.data_preset_cfg)

        assert self.img_backbone.name in ["resnet18", "resnet34", "resnet50", "HRNet"], "Wrong backbone for POEM"
        if self.img_backbone.name == "resnet18":
            self.feat_size = (512, 256, 128, 64)
        elif self.img_backbone.name == "resnet34":
            self.feat_size = (512, 256, 128, 64)
        elif self.img_backbone.name == "resnet50":
            self.feat_size = (2048, 1024, 512, 256)
        elif self.img_backbone.name == "HRNet":
            self.feat_size = (40, 80, 160, 320)

        if self.img_backbone.name != "HRNet":
            self.feat_delayer = nn.ModuleList([
                ConvBlock(self.feat_size[1] + self.feat_size[0], self.feat_size[1], kernel_size=3, relu=True,
                          norm='bn'),
                ConvBlock(self.feat_size[2] + self.feat_size[1], self.feat_size[2], kernel_size=3, relu=True,
                          norm='bn'),
                ConvBlock(self.feat_size[3] + self.feat_size[2], self.feat_size[3], kernel_size=3, relu=True,
                          norm='bn'),
            ])
            self.feat_in = ConvBlock(self.feat_size[3],
                                     self.feat_size[2],
                                     kernel_size=1,
                                     padding=0,
                                     relu=False,
                                     norm=None)

            # uv layer is used for extracting heatmaps (ResNet case)
            self.uv_delayer = nn.ModuleList([
                ConvBlock(self.feat_size[1] + self.feat_size[0], self.feat_size[1], kernel_size=3, relu=True,
                          norm='bn'),
                ConvBlock(self.feat_size[2] + self.feat_size[1], self.feat_size[2], kernel_size=3, relu=True,
                          norm='bn'),
                ConvBlock(self.feat_size[3] + self.feat_size[2], self.feat_size[3], kernel_size=3, relu=True,
                          norm='bn'),
            ])

            self.uv_out = ConvBlock(self.feat_size[3], self.num_joints, kernel_size=1, padding=0, relu=False, norm=None)
            self.uv_in = ConvBlock(self.num_joints, self.feat_size[2], kernel_size=1, padding=0, relu=True, norm='bn')
        else:
            # downsample the high-resolution representation by a 2-strided 3x3 convolution
            self.feat_delayer = nn.ModuleList([
                ConvBlock(self.feat_size[0], self.feat_size[1], kernel_size=3, stride=2, relu=True, norm='bn'),
                ConvBlock(self.feat_size[1], self.feat_size[2], kernel_size=3, stride=2, relu=True, norm='bn'),
                ConvBlock(self.feat_size[2], self.feat_size[3], kernel_size=3, stride=2, relu=True, norm='bn'),
            ])
            self.feat_in = ConvBlock(self.feat_size[3],
                                     self.feat_size[2],
                                     kernel_size=1,
                                     padding=0,
                                     relu=False,
                                     norm=None)

            # uv layer is used for extracting heatmaps (HRNet case)
            self.uv_delayer = nn.ModuleList([
                ConvBlock(self.feat_size[3] + self.feat_size[2], self.feat_size[2], kernel_size=3, relu=True,
                          norm='bn'),
                ConvBlock(self.feat_size[2] + self.feat_size[1], self.feat_size[1], kernel_size=3, relu=True,
                          norm='bn'),
                ConvBlock(self.feat_size[1] + self.feat_size[0], self.feat_size[0], kernel_size=3, relu=True,
                          norm='bn'),
            ])

            self.uv_out = ConvBlock(self.feat_size[0], self.num_joints, kernel_size=1, padding=0, relu=False, norm=None)
            self.uv_in = ConvBlock(self.num_joints, self.feat_size[1], kernel_size=1, padding=0, relu=True, norm='bn')

        self.ptEmb_head = build_head(cfg.HEAD, data_preset=self.data_preset_cfg)
        self.num_preds = self.ptEmb_head.num_preds

        self.mano_layer = ManoLayer(joint_rot_mode="axisang",
                                    use_pca=False,
                                    mano_assets_root="assets/mano_v1_2",
                                    center_idx=cfg.DATA_PRESET.CENTER_IDX,
                                    flat_hand_mean=True)

        self.face = self.mano_layer.th_faces

        self.joints_weight = cfg.LOSS.JOINTS_LOSS_WEIGHT
        self.vertices_weight = cfg.LOSS.VERTICES_LOSS_WEIGHT
        self.joints_2d_weight = cfg.LOSS.JOINTS_2D_LOSS_WEIGHT
        self.heatmap_joints_weights = cfg.LOSS.HEATMAP_JOINTS_WEIGHT
        self.vertices_2d_weight = cfg.LOSS.get("VERTICES_2D_LOSS_WEIGHT", 0.0)
        self.pose_weight = cfg.LOSS.get("POSE_LOSS_WEIGHT", 0.001)
        self.shape_weight = cfg.LOSS.get("SHAPE_LOSS_WEIGHT", 0.0005)

        if self.joints_loss_type == "l2":
            self.criterion_joints = torch.nn.MSELoss()
        else:
            self.criterion_joints = torch.nn.L1Loss()

        if self.verts_loss_type == "l2":
            self.criterion_vertices = torch.nn.MSELoss()
        else:
            self.criterion_vertices = torch.nn.L1Loss()

        if self.parametric_output:
            self.criterion_parameters = torch.nn.MSELoss()

        self.loss_metric = LossMetric(cfg)
        self.PA = PAEval(cfg, mesh_score=True)
        self.MPJPE_3D = MeanEPE(cfg, "joints_3d")
        self.MPJPE_3D_REF = MeanEPE(cfg, "joints_3d_ref")
        self.MPVPE_3D = MeanEPE(cfg, "vertices_3d")
        self.MPJPE_3D_REL = MeanEPE(cfg, "joints_3d_rel")
        self.MPVPE_3D_REL = MeanEPE(cfg, "vertices_3d_rel")
        self.MPTPE_3D = MeanEPE(cfg, "triangulate_joints")

        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL
        self.init_weights()

        logger.info(f"{self.name} has {param_size(self)}M parameters")
        logger.info(f"{self.name} loss type: joint {self.joints_loss_type} verts {self.verts_loss_type}")

    def init_weights(self):
        load_weights(self, pretrained=self.cfg.PRETRAINED)

    def setup(self, summary_writer, **kwargs):
        self.summary = summary_writer

    def feat_decode(self, mlvl_feats, backbone="HRNet"):
        if backbone != "HRNet":
            mlvl_feats_rev = list(reversed(mlvl_feats))
            x = mlvl_feats_rev[0]

            # There are some upsampling here
            # scale_factor=2: The factor by which the feature map is upsampled. In this case, the feature map is upsampled by a factor of 2.
            # Bilinear interpolation is used, which calculates the new pixel values based on the weighted average of the nearest four pixels.

            for i, fde in enumerate(self.feat_delayer):
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = torch.cat((x, mlvl_feats_rev[i + 1]), dim=1)
                x = fde(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)  # (BN, 64, 32, 32)
            x = self.feat_in(x)  # (BN, 128, 32, 32)

        else:
            x = mlvl_feats[0]

            for i, fde in enumerate(self.feat_delayer):
                interm = fde(x)
                x = interm + mlvl_feats[i + 1]

            # x := [BN, 320, 8, 8] prior to this line
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # x := [BN, 320, 16, 16]

            x = self.feat_in(x)  # (BN, 160, 16, 16)

        return x

    def uv_decode(self, mlvl_feats):
        # The following feat-dims are ResNet case / HRNet case
        mlvl_feats_rev = list(reversed(mlvl_feats))
        x = mlvl_feats_rev[0]
        for i, de in enumerate(self.uv_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat((x, mlvl_feats_rev[i + 1]), dim=1)
            x = de(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (BxN, 64, 32, 32) / (BxN, 40, 32, 32)
        uv_hmap = torch.sigmoid(self.uv_out(x))  # (BxN, 21, 32, 32) / (BxN, 21, 32, 32)
        uv_feat = self.uv_in(uv_hmap)  # (BxN, 128, 32, 32) / (BxN, 80, 32, 32)

        return uv_hmap, uv_feat

    def heatmap_stage(self, img_feats, W, H):
        # The following feat-dims are ResNet case
        uv_hmap, uv_feat = self.uv_decode(img_feats)  # (BN, 21, 32, 32), (BN, 128, 32, 32)
        uv_pdf = uv_hmap.reshape(*uv_hmap.shape[:2], -1)  # (BN, 21, 32x32)
        uv_pdf = uv_pdf / (uv_pdf.sum(dim=-1, keepdim=True) + 1e-6)  # Normalize the probability distributions
        uv_pdf = uv_pdf.contiguous().view(-1, self.num_joints, *uv_hmap.shape[-2:])  # (BN, 21, 32, 32)
        uv_coord = integral_heatmap2d(uv_pdf)  # (BN, 21, 2), range 0~1
        uv_coord_im = torch.einsum("bij, j->bij", uv_coord,
                                   torch.tensor([W, H]).to(
                                       uv_coord.device))  # scale the normalized coordinates back into range 0~W, H

        return uv_coord_im  # [BN, 21, 2]

    def extract_img_feat(self, img, backbone="HRNet"):
        # img := [BN, C, H, W]
        # Here the backbone features are fixed to be ResNet

        if backbone != "HRNet":
            img_feats = self.img_backbone(image=img)
            if isinstance(img_feats, dict):
                """img_feats for ResNet 34: 
                    torch.Size([BN, 64, 64, 64])
                    torch.Size([BN, 128, 32, 32])
                    torch.Size([BN, 256, 16, 16])
                    torch.Size([BN, 512, 8, 8])
                """
                img_feats = list([v for v in img_feats.values() if len(v.size()) == 4])
        else:
            """
                img_feats for HRNet:
                torch.Size([BN, 40, 64, 64])
                torch.Size([BN, 80, 32, 32])
                torch.Size([BN, 160, 16, 16])
                torch.Size([BN, 320, 8, 8])
            """
            img_feats = self.img_backbone(img)

        return img_feats

    def _forward_impl(self, batch, **kwargs):
        img = batch["image"]  # (BN, 3, H, W)

        img = img.view(-1, img.shape[-3], img.shape[-2], img.shape[-1])  # (BN, 3, H, W)
        batch_size = len(batch["cam_view_num"])
        inp_img_shape = img.shape[-2:]  # H, W
        H, W = inp_img_shape
        BN = img.shape[0]
        device = img.device
        mode = kwargs.get("mode", "train")

        # add hrnet backbone
        img_feats = self.extract_img_feat(img, self.img_backbone.name)

        # NOTE: @pengxiang add HRnet in merging multi-level features in backbone.
        mlvl_feat = self.feat_decode(img_feats, self.img_backbone.name)  # mlvl_feat := (BN, 128, 32, 32) for ResNet

        # This part is used for deriving the 2D joints (for training and inference)
        # We will use GT + noise for training, and use the predicted 2D joints for inference
        uv_coord_im_pred = self.heatmap_stage(img_feats, W, H)  # (BN, 21, 2)

        # This is used for the inference stage of single-view cases
        inputs_all_sv = (BN == batch_size)

        if mode == "train":
            ref_joints = batch["master_joints_3d"].reshape(-1, 21, 3)
            noise_3d = 0.01 * torch.randn(batch_size, 21, 3).to(device)
            noise_3d += 0.01 * torch.randn(1).to(device)
            ref_joints = ref_joints + noise_3d
            ref_joints_root_noise = ref_joints[:, self.center_idx, :].unsqueeze(1)
            noise_scale = (0.01 * (torch.rand(1) * 2 - 1) + 1.0).to(device)
            ref_joints = noise_scale * (ref_joints - ref_joints_root_noise) + ref_joints_root_noise
        elif inputs_all_sv:
            ref_joints = batch["master_joints_3d"].reshape(-1, 21, 3)
        else:
            uv_coord_im = uv_coord_im_pred
            K = batch['target_cam_intr'].view(-1, 3, 3)  # (BN, 3, 3)
            T_c2m = torch.linalg.inv(batch['target_cam_extr'].view(-1, 4, 4))  # (BN, 4, 4)

            # Iterate over each subbatch for DLT under different num of cam views
            ref_joints = []
            for i in range(batch_size):
                start_idx = np.sum(batch["cam_view_num"][:i])
                end_idx = np.sum(batch["cam_view_num"][:i + 1])
                uv_coord_im_sub = uv_coord_im[start_idx:end_idx].unsqueeze(0)
                K_sub = K[start_idx:end_idx].unsqueeze(0)
                T_c2m_sub = T_c2m[start_idx:end_idx].unsqueeze(0)
                ref_joints_sub = batch_triangulate_dlt_torch(uv_coord_im_sub, K_sub, T_c2m_sub)
                ref_joints.append(ref_joints_sub)
            ref_joints = torch.cat(ref_joints, dim=0)

        gt_J3d = batch["master_joints_3d"].reshape(-1, 21, 3)
        gt_V3d = batch["master_verts_3d"].reshape(-1, 778, 3)
        gt_mesh = torch.cat([gt_J3d, gt_V3d], dim=1)  # (B, 799, 3)

        # prepare image_metas
        img_metas = {
            "inp_img_shape": inp_img_shape,  # h, w
            "cam_intr": batch["target_cam_intr"].reshape(-1, 3, 3),  # tensor (BN, 3, 3)
            "cam_extr": batch["target_cam_extr"].reshape(-1, 4, 4),  # tensor  (BN, 4, 4)
            "master_id": batch["master_id"],  # lst (B, )
            "ref_mesh_gt": gt_mesh,
            "cam_view_num": batch["cam_view_num"]
        }

        debug_metas = {"img": batch["image"], "2d_joints_gt": batch["target_joints_2d"]}

        preds = self.ptEmb_head(mlvl_feat=mlvl_feat,
                                img_metas=img_metas,
                                reference_joints=ref_joints,
                                debug_metas=debug_metas)

        # last decoder's output
        pred_joints_3d = preds["all_coords_preds"][-1, :, :self.num_joints, :]  # (B, 21, 3)
        pred_verts_3d = preds["all_coords_preds"][-1, :, self.num_joints:, :]  # (B, 778, 3)

        preds["pred_joints_3d"] = pred_joints_3d
        preds["pred_verts_3d"] = pred_verts_3d
        center_joint = pred_joints_3d[:, self.center_idx, :].unsqueeze(1)  # (B, 1, 3)
        preds["pred_joints_3d_rel"] = pred_joints_3d - center_joint
        preds["pred_verts_3d_rel"] = pred_verts_3d - center_joint
        preds["pred_joints_uv"] = uv_coord_im_pred  # (BN, 21, 2)
        preds["pred_ref_joints_3d"] = ref_joints  # (B, 21, 3)
        return preds

    @staticmethod
    def loss_proj_to_multicam(pred_joints, T_c2m, K, gt_joints_2d, cam_view_num, img_scale):
        """
            pred_joints := [B, 21, 3]
            T_c2m := [BN, 4, 4]
            K := [BN, 3, 3]
            gt_joints_2d := [BN, 21, 2]
            cam_view_num := [List] of cam view numbers
            img_scale: float
        """
        batch_size = len(cam_view_num)
        pred_joints_2d = []
        for i in range(batch_size):
            start_idx = np.sum(cam_view_num[:i])
            end_idx = np.sum(cam_view_num[:i + 1])

            pred_joints_sub = pred_joints[i].unsqueeze(0).repeat(1, cam_view_num[i], 1, 1)
            pred_joints_in_cam_sub = batch_cam_extr_transf(T_c2m[start_idx:end_idx].unsqueeze(0), pred_joints_sub)
            pred_joints_2d_sub = batch_cam_intr_projection(K[start_idx:end_idx].unsqueeze(0), pred_joints_in_cam_sub)
            pred_joints_2d.append(pred_joints_2d_sub)
        pred_joints_2d = torch.concat(pred_joints_2d, dim=1).squeeze(0)  # (BN, 21, 2)

        multicam_proj_offset = torch.clamp(pred_joints_2d - gt_joints_2d, min=-.5 * img_scale,
                                           max=.5 * img_scale) / img_scale
        loss_2d_joints = torch.sum(torch.pow(multicam_proj_offset, 2), dim=2)  # (BN, 21, 2)
        loss_2d_joints = torch.mean(loss_2d_joints)
        return loss_2d_joints

    def compute_loss(self, preds, gt):
        all_coords_preds = preds["all_coords_preds"]  # (N_Decoder, B, NUM_QUERY, 3)
        loss_dict = {}
        loss = 0

        batch_size = len(gt["cam_view_num"])
        H = gt["image"].size(-2)
        W = gt["image"].size(-1)
        # use diagonal as scale
        img_scale = math.sqrt(float(W**2 + H**2))
        master_joints_gt = gt["master_joints_3d"].view(-1, 21, 3)  # (B, 21, 3)
        master_verts_gt = gt["master_verts_3d"].view(-1, 778, 3)  # (B, 778, 3)

        # Here we have added back the heatmap loss
        loss_heatmap_joints = (preds["pred_joints_uv"] - gt["target_joints_2d"]) / img_scale
        loss_heatmap_joints = torch.sum(torch.pow(loss_heatmap_joints, 2), dim=2)  # (BN, 21, 2)
        loss_heatmap_joints = torch.mean(loss_heatmap_joints)
        loss_dict["loss_heatmap_joints"] = loss_heatmap_joints
        loss += self.heatmap_joints_weights * loss_heatmap_joints

        gt_T_c2m = torch.linalg.inv(gt["target_cam_extr"].view(-1, 4, 4))  # (BN , 4, 4)
        pred_joints = all_coords_preds[-1, :, :self.num_joints, :]  # (B, 21, 3)
        pred_verts = all_coords_preds[-1, :, self.num_joints:, :]  # (B, 778, 3)
        pred_joints_from_mesh = mano_to_openpose(self.mano_layer.th_J_regressor, pred_verts)
        gt_joints_from_mesh = mano_to_openpose(self.mano_layer.th_J_regressor, master_verts_gt)

        # Use iteration for random cam views
        gt_verts_2d_ncams = []
        for j in range(batch_size):
            start_idx = np.sum(gt["cam_view_num"][:j])
            end_idx = np.sum(gt["cam_view_num"][:j + 1])

            gt_verts_ncams_sub = master_verts_gt[j].unsqueeze(0).repeat(1, gt["cam_view_num"][j], 1, 1)
            gt_verts_ncams_sub = batch_cam_extr_transf(gt_T_c2m[start_idx:end_idx].unsqueeze(0), gt_verts_ncams_sub)
            gt_verts_2d_ncams_sub = batch_cam_intr_projection(gt["target_cam_intr"][start_idx:end_idx].unsqueeze(0),
                                                              gt_verts_ncams_sub)
            gt_verts_2d_ncams.append(gt_verts_2d_ncams_sub)
        gt_verts_2d_ncams = torch.concat(gt_verts_2d_ncams, dim=1).squeeze(0)  # (BN, 778, 2)

        # 3D joints loss
        loss_3d_joints_from_mesh = self.criterion_joints(pred_joints_from_mesh, gt_joints_from_mesh)
        loss_3d_joints = self.criterion_joints(pred_joints, master_joints_gt)
        loss_recon = self.joints_weight * (loss_3d_joints + loss_3d_joints_from_mesh)

        # 3D verts loss
        if self.parametric_output:
            # ! IMPORTANT. self.transformer_center_idx = 9 while self.center_idx = 0
            # RR_V / RR_J is calculated with root_idx = 0, but here the loss is calculated with root_idx = 9
            # This is used to align the output of the transformer
            # For parametric output, we have set the root_idx = 9
            center_joint_gt = master_joints_gt[:, self.transformer_center_idx, :].unsqueeze(1)  # (B, 1, 3)
            pred_verts_rel = pred_verts - center_joint_gt
            gt_verts_rel = master_verts_gt - center_joint_gt
            loss_3d_verts = self.criterion_vertices(pred_verts_rel, gt_verts_rel)
        else:
            loss_3d_verts = self.criterion_vertices(pred_verts, master_verts_gt)
        loss_recon += self.vertices_weight * loss_3d_verts

        # 2D joints loss
        if self.joints_2d_weight != 0:
            loss_2d_joints = self.loss_proj_to_multicam(pred_joints, gt_T_c2m, gt["target_cam_intr"],
                                                        gt["target_joints_2d"], gt["cam_view_num"], img_scale)
        else:
            loss_2d_joints = torch.tensor(0.0).float().to(pred_verts.device)
        loss_recon += self.joints_2d_weight * loss_2d_joints

        # 2D verts loss
        if self.vertices_2d_weight != 0:
            loss_2d_verts = self.loss_proj_to_multicam(pred_verts, gt_T_c2m, gt["target_cam_intr"], gt_verts_2d_ncams,
                                                       gt["cam_view_num"], img_scale)
        else:
            loss_2d_verts = torch.tensor(0.0).float().to(pred_verts.device)
        loss_recon += self.vertices_2d_weight * loss_2d_verts

        # parametric loss
        if self.parametric_output:
            master_index = [np.sum(gt["cam_view_num"][:j]) for j in range(len(gt["cam_view_num"]))]
            # use torch split to get the master_mano_pose and mano_shape
            master_mano_pose = gt["mano_pose"][master_index]
            master_mano_shape = gt["mano_shape"][master_index]
            loss_pose = self.criterion_parameters(preds["pred_pose"], master_mano_pose)
            loss_shape = self.criterion_parameters(preds["pred_shape"], master_mano_shape)
        else:
            loss_pose = torch.tensor(0.0).float().to(pred_verts.device)
            loss_shape = torch.tensor(0.0).float().to(pred_verts.device)
        loss_recon += self.pose_weight * loss_pose + self.shape_weight * loss_shape

        # loss += loss_recon
        loss_dict[f'loss_3d_joints'] = loss_3d_joints
        loss_dict[f'loss_3d_joints_from_mesh'] = loss_3d_joints_from_mesh
        loss_dict[f'loss_3d_verts'] = loss_3d_verts
        loss_dict[f'loss_recon'] = loss_recon
        loss += loss_recon

        if self.joints_2d_weight != 0:
            loss_dict[f'loss_2d_joints'] = loss_2d_joints
        if self.vertices_2d_weight != 0:
            loss_dict[f"loss_2d_verts"] = loss_2d_verts
        if self.parametric_output:
            loss_dict[f"loss_pose"] = loss_pose
            loss_dict[f"loss_shape"] = loss_shape

        loss_dict['loss'] = loss
        return loss, loss_dict

    def training_step(self, batch, step_idx, **kwargs):
        img = batch["image"]  # (B, N, 3, H, W) 5 channels
        batch_size = len(batch["cam_view_num"])

        master_joints_3d = batch["master_joints_3d"].view(-1, 21, 3)  # (B, 21, 3)
        master_verts_3d = batch["master_verts_3d"].view(-1, 778, 3)  # (B, 778, 3)

        preds = self._forward_impl(batch, mode="train", **kwargs)
        loss, loss_dict = self.compute_loss(preds, batch)

        ## last decoder's output
        pred_joints_3d = preds["pred_joints_3d"]  # (B, 21, 3)
        pred_verts_3d = preds["pred_verts_3d"]
        self.MPJPE_3D.feed(pred_joints_3d, gt_kp=master_joints_3d)
        self.MPVPE_3D.feed(pred_verts_3d, gt_kp=master_verts_3d)
        self.loss_metric.feed(loss_dict, batch_size)

        if step_idx % self.train_log_interval == 0:
            for k, v in loss_dict.items():
                self.summary.add_scalar(f"{k}", v.item(), step_idx)
            self.summary.add_scalar("MPJPE_3D", self.MPJPE_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_3D", self.MPVPE_3D.get_result(), step_idx)

            if step_idx % (self.train_log_interval * 5) == 0:  # viz every 10 * interval batches
                # Visualization of the first image and its first view
                img_toshow = img[0, ...].unsqueeze(0)  # (1, 3, H, W)
                extr_toshow = torch.linalg.inv(batch["target_cam_extr"][0, ...]).unsqueeze(0)  # (1, 4, 4)
                intr_toshow = batch["target_cam_intr"][0, ...].unsqueeze(0)  # (1, 3, 3)

                pred_verts_3d_single = pred_verts_3d[0].unsqueeze(0)  # (1, 778, 3)
                master_verts_3d_single = master_verts_3d[0].unsqueeze(0)

                pred_V3d_in_cam = (extr_toshow[:, :3, :3] @ pred_verts_3d_single.transpose(1, 2)).transpose(1, 2)
                pred_V3d_in_cam = pred_V3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)
                gt_V3d_in_cam = (extr_toshow[:, :3, :3] @ master_verts_3d_single.transpose(1, 2)).transpose(1, 2)
                gt_V3d_in_cam = gt_V3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)

                # Visualization of heatmap regression for one image.
                pred_uv_coords = preds['pred_joints_uv'][0].unsqueeze(0)
                gt_uv_coords = batch["target_joints_2d"][0].unsqueeze(0)
                img_array_joints = draw_batch_joint_images(pred_uv_coords, gt_uv_coords, img_toshow, step_idx)
                self.summary.add_image(f"img/viz_joints_2d_train", img_array_joints, step_idx, dataformats="NHWC")

                pred_V2d = batch_persp_project(pred_V3d_in_cam, intr_toshow)  # (B, 21, 2)
                gt_V2d = batch_persp_project(gt_V3d_in_cam, intr_toshow)  # (B, 21, 2)
                img_array_verts = draw_batch_verts_images(pred_V2d, gt_V2d, img_toshow, step_idx)
                self.summary.add_image(f"img/viz_verts_2d_train", img_array_verts, step_idx, dataformats="NHWC")

        return preds, loss_dict

    def on_train_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        recorder.record_metric([self.MPJPE_3D, self.MPVPE_3D], epoch_idx, comment=comment)
        self.loss_metric.reset()
        self.MPJPE_3D.reset()
        self.MPVPE_3D.reset()

    def validation_step(self, batch, step_idx, **kwargs):
        preds = self.testing_step(batch, step_idx, **kwargs)
        img = batch["image"]  # (BN, 3, H, W) 4 channels
        batch_size = len(batch["cam_view_num"])
        min_views = min(batch["cam_view_num"])  # Min-camera views within the batch
        pred_joints_3d = preds["pred_joints_3d"]  # (B, 21, 3)
        master_joints_gt = batch["master_joints_3d"].view(batch_size, 21, 3)  # (B, 21, 3)

        pred_verts_3d = preds["pred_verts_3d"]
        master_verts_3d = batch["master_verts_3d"].view(batch_size, 778, 3)  # (B, 778, 3)

        self.summary.add_scalar("MPJPE_3D_val", self.MPJPE_3D.get_result(), step_idx)
        self.summary.add_scalar("MPVPE_3D_val", self.MPVPE_3D.get_result(), step_idx)

        if step_idx % (self.train_log_interval * 10) == 0:
            view_id = np.random.randint(min_views)
            idx_list = [view_id + np.sum(batch["cam_view_num"][:j]) for j in range(batch_size)]
            img_toshow = img[idx_list, ...]  # (B, 3, H, W)
            extr_toshow = torch.linalg.inv(batch["target_cam_extr"][idx_list, ...])  # (B, 4, 4)
            intr_toshow = batch["target_cam_intr"][idx_list, ...]  # (B, 3, 3)

            # Only keep the vertices in this stage
            pred_V3d_in_cam = (extr_toshow[:, :3, :3] @ pred_verts_3d.transpose(1, 2)).transpose(1, 2)
            pred_V3d_in_cam = pred_V3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)
            gt_V3d_in_cam = (extr_toshow[:, :3, :3] @ master_verts_3d.transpose(1, 2)).transpose(1, 2)
            gt_V3d_in_cam = gt_V3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)

            pred_uv_coords = preds['pred_joints_uv'][0].unsqueeze(0)
            gt_uv_coords = batch["target_joints_2d"][0].unsqueeze(0)
            img_array_joints = draw_batch_joint_images(pred_uv_coords, gt_uv_coords, img_toshow, step_idx)
            self.summary.add_image(f"img/viz_joints_2d_val", img_array_joints, step_idx, dataformats="NHWC")

            pred_V2d = batch_persp_project(pred_V3d_in_cam, intr_toshow)  # (B, 21, 2)
            gt_V2d = batch_persp_project(gt_V3d_in_cam, intr_toshow)  # (B, 21, 2)
            img_array_verts = draw_batch_verts_images(pred_V2d, gt_V2d, img_toshow, step_idx)
            self.summary.add_image(f"img/viz_verts_2d_val", img_array_verts, step_idx, dataformats="NHWC")

        return preds

    def on_val_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val"
        recorder.record_metric([
            self.MPJPE_3D,
            self.MPJPE_3D_REF,
            self.MPVPE_3D,
            self.MPJPE_3D_REL,
            self.MPVPE_3D_REL,
            self.PA,
            self.MPTPE_3D,
        ],
                               epoch_idx,
                               comment=comment)
        self.MPJPE_3D.reset()
        self.MPVPE_3D.reset()
        self.MPJPE_3D_REF.reset()
        self.MPJPE_3D_REL.reset()
        self.MPVPE_3D_REL.reset()
        self.PA.reset()
        self.MPTPE_3D.reset()

    def testing_step(self, batch, step_idx, **kwargs):
        img = batch["image"]  # (BN, 3, H, W) 4 channels
        batch_size = len(batch["cam_view_num"])

        preds = self._forward_impl(batch, mode="test", **kwargs)
        pred_ref_joints_3d = preds["pred_ref_joints_3d"]  # (B, 21, 3)
        master_joints_3d = batch["master_joints_3d"].view(batch_size, 21, 3)  # (B, 21, 3)
        self.MPTPE_3D.feed(pred_ref_joints_3d, gt_kp=master_joints_3d)

        master_verts_3d = batch["master_verts_3d"].view(batch_size, 778, 3)  # (B, 778, 3)
        pred_verts_3d = preds["pred_verts_3d"]

        if self.pred_joints_from_mesh is False:
            master_joints_3d = batch["master_joints_3d"]  # (B, 21, 3)
            pred_joints_3d = preds["pred_joints_3d"]  # (B, 21, 3)
        else:
            master_joints_3d = mano_to_openpose(self.mano_layer.th_J_regressor, master_verts_3d)
            pred_joints_3d = mano_to_openpose(self.mano_layer.th_J_regressor, pred_verts_3d)

        pred_ref_J3d = pred_ref_joints_3d
        pred_J3d = pred_joints_3d
        pred_V3d = pred_verts_3d
        gt_J3d = master_joints_3d
        gt_V3d = master_verts_3d

        pred_J3d_rel = pred_J3d - pred_J3d[:, self.center_idx, :].unsqueeze(1)
        pred_V3d_rel = pred_V3d - pred_J3d[:, self.center_idx, :].unsqueeze(1)
        gt_J3d_rel = gt_J3d - gt_J3d[:, self.center_idx, :].unsqueeze(1)
        gt_V3d_rel = gt_V3d - gt_J3d[:, self.center_idx, :].unsqueeze(1)

        # Calculate the loss in master space
        self.MPJPE_3D.feed(pred_J3d, gt_kp=gt_J3d)
        self.MPJPE_3D_REF.feed(pred_ref_J3d, gt_kp=gt_J3d)
        self.MPVPE_3D.feed(pred_V3d, gt_kp=gt_V3d)

        self.MPJPE_3D_REL.feed(pred_J3d_rel, gt_kp=gt_J3d_rel)
        self.MPVPE_3D_REL.feed(pred_V3d_rel, gt_kp=gt_V3d_rel)

        self.PA.feed(pred_J3d, gt_J3d, pred_V3d, gt_V3d)

        if "callback" in kwargs:
            callback = kwargs.pop("callback")
            if callable(callback):
                callback(preds, batch, step_idx, **kwargs)

        return preds

    def on_test_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-test"
        recorder.record_metric([
            self.MPJPE_3D,
            self.MPJPE_3D_REF,
            self.MPVPE_3D,
            self.MPJPE_3D_REL,
            self.MPVPE_3D_REL,
            self.PA,
            self.MPTPE_3D,
        ],
                               epoch_idx,
                               comment=comment)
        self.MPJPE_3D.reset()
        self.MPJPE_3D_REF.reset()
        self.MPVPE_3D.reset()
        self.MPJPE_3D_REL.reset()
        self.MPVPE_3D_REL.reset()
        self.PA.reset()
        self.MPTPE_3D.reset()

    def format_metric(self, mode="train"):
        if mode == "train":
            if self.parametric_output:
                return (f"L: {self.loss_metric.get_loss('loss'):.4f} | "
                        f"L_V: {self.loss_metric.get_loss('loss_3d_verts'):.4f} | "
                        f"L_J: {self.loss_metric.get_loss('loss_3d_joints'):.4f} | "
                        f"L_P: {self.loss_metric.get_loss('loss_pose'):.2f} | "
                        f"L_S: {self.loss_metric.get_loss('loss_shape'):.2f} | ")
            else:
                return (f"L: {self.loss_metric.get_loss('loss'):.4f} | "
                        f"L_V: {self.loss_metric.get_loss('loss_3d_verts'):.4f} | "
                        f"L_J: {self.loss_metric.get_loss('loss_heatmap_joints'):.4f} | ")
        elif mode == "test":
            metric_toshow = [self.PA, self.MPJPE_3D_REF]
        else:
            metric_toshow = [self.MPJPE_3D, self.MPVPE_3D]

        return " | ".join([str(me) for me in metric_toshow])

    def forward(self, inputs, step_idx, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx, **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, step_idx, **kwargs)
        elif mode == "test":
            return self.testing_step(inputs, step_idx, **kwargs)
        elif mode == "inference":
            return self.inference_step(inputs, step_idx, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")
