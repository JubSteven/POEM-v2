import os
import pickle

import numpy as np
import torch
from manotorch.manolayer import ManoLayer

from lib.metrics.pck import Joint3DPCK, Vert3DPCK
from lib.utils.transform import (batch_cam_extr_transf, batch_cam_intr_projection, denormalize)
from lib.viztools.draw import save_a_image_with_mesh_joints
from lib.viztools.opendr_renderer import OpenDRRenderer

from .logger import logger


class IdleCallback():

    def __init__(self):
        pass

    def __call__(self, preds, inputs, step_idx, **kwargs):
        pass

    def on_finished(self):
        pass

    def reset(self):
        pass


class AUCCallback(IdleCallback):

    def __init__(self, exp_dir, val_min=0.0, val_max=0.02, steps=20):
        self.exp_dir = exp_dir
        self.val_min = val_min
        self.val_max = val_max
        self.steps = steps
        self.PCK_J = Joint3DPCK(EVAL_TYPE="joints_3d", VAL_MIN=val_min, VAL_MAX=val_max, STEPS=steps)
        self.PCK_V = Vert3DPCK(EVAL_TYPE="verts_3d", VAL_MIN=val_min, VAL_MAX=val_max, STEPS=steps)

    def reset(self):
        self.PCK_J.reset()
        self.PCK_V.reset()

    def __call__(self, preds, inputs, step_idx, **kwargs):
        self.PCK_J.feed(preds, inputs)
        self.PCK_V.feed(preds, inputs)

    def on_finished(self):

        logger.info(f"Dump AUC results to {self.exp_dir}")
        filepth_j = os.path.join(self.exp_dir, 'res_auc_j.pkl')
        auc_pth_j = os.path.join(self.exp_dir, 'auc_j.txt')
        filepth_v = os.path.join(self.exp_dir, 'res_auc_v.pkl')
        auc_pth_v = os.path.join(self.exp_dir, 'auc_v.txt')

        dict_J = self.PCK_J.get_measures()
        dict_V = self.PCK_V.get_measures()

        with open(filepth_j, 'wb') as f:
            pickle.dump(dict_J, f)
        with open(auc_pth_j, 'w') as ff:
            ff.write(str(dict_J["auc_all"]))

        with open(filepth_v, 'wb') as f:
            pickle.dump(dict_V, f)
        with open(auc_pth_v, 'w') as ff:
            ff.write(str(dict_V["auc_all"]))

        logger.warning(f"auc_j: {dict_J['auc_all']}")
        logger.warning(f"auc_v: {dict_V['auc_all']}")
        self.reset()


class PredictionSaverCallback(IdleCallback):

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        # Ensure preds directory exists
        preds_dir = os.path.join(self.exp_dir, 'preds')
        os.makedirs(preds_dir, exist_ok=True)

    def __call__(self, preds, inputs, step_idx):
        # Integrate preds and input data into one dictionary
        data_to_save = {
            "pred_joints_3d": preds["pred_joints_3d"].detach().cpu().numpy(),
            "pred_verts_3d": preds["pred_verts_3d"].detach().cpu().numpy(),
            "target_cam_extr": inputs["target_cam_extr"].detach().cpu().numpy(),
            "target_cam_intr": inputs["target_cam_intr"].detach().cpu().numpy(),
            "cam_extr": inputs["cam_extr"].detach().cpu().numpy(),
            "cam_intr": inputs["cam_intr"].detach().cpu().numpy(),
            "_hand_side_ori": inputs["_hand_side_ori"],
        }

        preds_dir = os.path.join(self.exp_dir, 'preds')
        preds_filepath = os.path.join(preds_dir, f"{step_idx}.pkl")
        with open(preds_filepath, 'wb') as f:
            pickle.dump(data_to_save, f)


class DrawingHandCallback(IdleCallback):

    def __init__(self, img_draw_dir):

        self.img_draw_dir = img_draw_dir
        os.makedirs(img_draw_dir, exist_ok=True)

        mano_layer = ManoLayer(mano_assets_root="assets/mano_v1_2")
        self.mano_faces = mano_layer.get_mano_closed_faces().numpy()
        self.renderer = OpenDRRenderer()

    def __call__(self, preds, inputs, step_idx, **kwargs):
        tensor_image = inputs["image"]  # (BN, 3, H, W) 4 channels
        cam_num = inputs["cam_view_num"]
        cam_intr = inputs["target_cam_intr"]
        cam_extr = inputs["target_cam_extr"]

        gt_vts = inputs["master_verts_3d"].reshape(2, 778, 3)
        gt_jts = inputs["master_joints_3d"].reshape(2, 21, 3)

        batch_size = len(cam_num)
        for i in range(batch_size):
            start_idx = np.sum(cam_num[:i])
            end_idx = np.sum(cam_num[:i + 1])
            tensor_image_sub = tensor_image[start_idx:end_idx].unsqueeze(0)
            n_views = tensor_image_sub.size(1)
            image = denormalize(tensor_image_sub, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False)
            image = image.permute(0, 1, 3, 4, 2)
            image = image.mul_(255.0).detach().cpu()  # (BN, H, W, 3)
            image = image.numpy().astype(np.uint8)

            mesh_xyz = preds["pred_verts_3d"][i].unsqueeze(0).repeat(cam_num[i], 1, 1)  # (N, 778, 3)
            pose_xyz = preds["pred_joints_3d"][i].unsqueeze(0).repeat(cam_num[i], 1, 1)  # (N, 21, 3)

            # GT
            device = preds['pred_verts_3d'][i].device
            gt_mesh_xyz = gt_vts[i].to(device).unsqueeze(0).repeat(cam_num[i], 1, 1)  # (N, 778, 3)
            gt_pose_xyz = gt_jts[i].to(device).unsqueeze(0).repeat(cam_num[i], 1, 1)  # (N, 21, 3)

            gt_mesh_xyz = gt_mesh_xyz.unsqueeze(0)
            gt_pose_xyz = gt_pose_xyz.unsqueeze(0)

            cam_param = cam_intr[start_idx:end_idx].to(mesh_xyz.device)  # (N, 3, 3)
            gt_T_c2m = torch.linalg.inv(cam_extr[start_idx:end_idx]).to(mesh_xyz.device)  # (N, 4, 4)

            # Add a dummy dimension to apply the following batch operations
            mesh_xyz = mesh_xyz.unsqueeze(0)
            pose_xyz = pose_xyz.unsqueeze(0)
            cam_param = cam_param.unsqueeze(0)
            gt_T_c2m = gt_T_c2m.unsqueeze(0)

            mesh_xyz = batch_cam_extr_transf(gt_T_c2m, mesh_xyz)  # (1, N, 21, 3)
            pose_xyz = batch_cam_extr_transf(gt_T_c2m, pose_xyz)  # (1, N, 778, 3)
            pose_uv = batch_cam_intr_projection(cam_param, pose_xyz)  # (1, N, 21, 2)

            gt_mesh_xyz = batch_cam_extr_transf(gt_T_c2m, gt_mesh_xyz)  # (1, N, 778, 3)
            gt_pose_xyz = batch_cam_extr_transf(gt_T_c2m, gt_pose_xyz)  # (1, N, 21, 3)
            gt_pose_uv = batch_cam_intr_projection(cam_param, gt_pose_xyz)  # (1, N, 21, 2)

            mesh_xyz = mesh_xyz.detach().cpu().numpy()
            pose_xyz = pose_xyz.detach().cpu().numpy()
            pose_uv = pose_uv.detach().cpu().numpy()
            cam_param = cam_param.detach().cpu().numpy()

            gt_mesh_xyz = gt_mesh_xyz.detach().cpu().numpy()
            gt_pose_xyz = gt_pose_xyz.detach().cpu().numpy()
            gt_pose_uv = gt_pose_uv.detach().cpu().numpy()

            for j in range(n_views):
                file_name = os.path.join(self.img_draw_dir, f"step{step_idx}_frame{i}_view{j}.jpg")
                save_a_image_with_mesh_joints(image=image[0, j],
                                              cam_param=cam_param[0, j],
                                              mesh_xyz=mesh_xyz[0, j],
                                              pose_uv=pose_uv[0, j],
                                              pose_xyz=pose_xyz[0, j],
                                              face=self.mano_faces,
                                              with_mayavi_mesh=False,
                                              with_skeleton_3d=False,
                                              file_name=file_name,
                                              renderer=self.renderer)

                file_name = os.path.join(self.img_draw_dir, f"step{step_idx}_frame{i}_view{j}_GT.jpg")
                save_a_image_with_mesh_joints(image=image[0, j],
                                              cam_param=cam_param[0, j],
                                              mesh_xyz=gt_mesh_xyz[0, j],
                                              pose_uv=gt_pose_uv[0, j],
                                              pose_xyz=gt_pose_xyz[0, j],
                                              face=self.mano_faces,
                                              with_mayavi_mesh=False,
                                              with_skeleton_3d=False,
                                              file_name=file_name,
                                              renderer=self.renderer)

    def on_finished(self):
        pass
