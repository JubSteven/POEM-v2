import os
import pickle

import numpy as np
import torch
from manotorch.manolayer import ManoLayer

from lib.metrics.pck import Joint3DPCK, Vert3DPCK
from lib.utils.transform import (batch_cam_extr_transf, batch_cam_intr_projection, denormalize)
from lib.viztools.draw import save_a_image_with_mesh_joints
from lib.viztools.opendr_renderer import OpenDRRenderer
from lib.datasets.ho3d_official_test import HO3DOfficialTestEvalUtil

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
        self.preds_dir = preds_dir

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

        preds_filepath = os.path.join(self.preds_dir, f"{step_idx}.pkl")
        with open(preds_filepath, 'wb') as f:
            pickle.dump(data_to_save, f)


class HO3DOfficialEvalCallback(IdleCallback):

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        # Ensure preds directory exists
        preds_dir = os.path.join(self.exp_dir, 'preds')
        os.makedirs(preds_dir, exist_ok=True)
        self.preds_dir = preds_dir

        # init eval utils
        self.eval_xyz = HO3DOfficialTestEvalUtil()
        self.eval_xyz_procrustes_aligned = HO3DOfficialTestEvalUtil()
        self.eval_xyz_sc_tr_aligned = HO3DOfficialTestEvalUtil()
        self.eval_mesh_err = HO3DOfficialTestEvalUtil(num_kp=778)
        self.eval_mesh_err_aligned = HO3DOfficialTestEvalUtil(num_kp=778)
        self.f_score, self.f_score_aligned = list(), list()
        self.f_threshs = [0.005, 0.015]
        self.n_eval_samples = 0

    @staticmethod
    def calculate_fscore(gt, pr, th=0.01):
        d1 = np.min(np.linalg.norm(np.expand_dims(gt, axis=-2) - np.expand_dims(pr, axis=-3), axis=-1), axis=-1)
        d2 = np.min(np.linalg.norm(np.expand_dims(pr, axis=-2) - np.expand_dims(gt, axis=-3), axis=-1), axis=-1)
        if len(d1) and len(d2):
            recall = float(sum(d < th for d in d2)) / float(
                len(d2))  # how many of our predicted points lie close to a gt point?
            precision = float(sum(d < th for d in d1)) / float(len(d1))  # how many of gt points are matched?

            if recall + precision > 0:
                fscore = 2 * recall * precision / (recall + precision)
            else:
                fscore = 0
        else:
            fscore = 0
            precision = 0
            recall = 0
        return fscore, precision, recall

    @staticmethod
    def align_sc_tr(mtx1, mtx2):
        """ Align the 3D joint location with the ground truth by scaling and translation """

        predCurr = mtx2.copy()
        # normalize the predictions
        s = np.sqrt(np.sum(np.square(predCurr[4] - predCurr[0])))
        if s > 0:
            predCurr = predCurr / s

        # get the scale of the ground truth
        sGT = np.sqrt(np.sum(np.square(mtx1[4] - mtx1[0])))

        # make predictions scale same as ground truth scale
        predCurr = predCurr * sGT

        # make preditions translation of the wrist joint same as ground truth
        predCurrRel = predCurr - predCurr[0:1, :]
        preds_sc_tr_al = predCurrRel + mtx1[0:1, :]

        return preds_sc_tr_al

    @staticmethod
    def align_w_scale(mtx1, mtx2, return_trafo=False):
        """ Align the predicted entity in some optimality sense with the ground truth. """
        # center
        t1 = mtx1.mean(0)
        t2 = mtx2.mean(0)
        mtx1_t = mtx1 - t1
        mtx2_t = mtx2 - t2

        # scale
        s1 = np.linalg.norm(mtx1_t) + 1e-8
        mtx1_t /= s1
        s2 = np.linalg.norm(mtx2_t) + 1e-8
        mtx2_t /= s2

        # orth alignment
        from scipy.linalg import orthogonal_procrustes
        R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

        # apply trafos to the second matrix
        mtx2_t = np.dot(mtx2_t, R.T) * s
        mtx2_t = mtx2_t * s1 + t1
        if return_trafo:
            return R, s, s1, t1 - t2
        else:
            return mtx2_t

    @staticmethod
    def align_by_trafo(mtx, trafo):
        t2 = mtx.mean(0)
        mtx_t = mtx - t2
        R, s, s1, t1 = trafo
        return np.dot(mtx_t, R.T) * s * s1 + t1 + t2

    def __call__(self, preds, inputs, step_idx, **kwargs):
        cam_num = inputs["cam_view_num"]
        batch_size = len(cam_num)

        batch_xyz = inputs["batch_xyz"].cpu().numpy()
        batch_xyz_pred = preds["batch_xyz_pred"].cpu().numpy()  # (B, 21, 3)
        batch_verts = inputs["batch_verts"].cpu().numpy()
        batch_verts_pred = preds["batch_verts_pred"].cpu().numpy()  # (B, 778, 3)

        for i in range(batch_size):
            xyz = batch_xyz[i]  # (21, 3)
            xyz_pred = batch_xyz_pred[i]  # (21, 3)
            verts = batch_verts[i]
            verts_pred = batch_verts_pred[i]

            n_cam = cam_num[i]
            for j in range(n_cam):
                # if n_cams, this error should be weighted by the number of cameras
                self.eval_xyz.feed(xyz, np.ones_like(xyz[:, 0]), xyz_pred)
                self.eval_mesh_err.feed(verts, np.ones_like(verts[:, 0]), verts_pred)

                # scale and translation aligned predictions for xyz
                xyz_pred_sc_tr_aligned = self.align_sc_tr(xyz, xyz_pred)
                self.eval_xyz_sc_tr_aligned.feed(xyz, np.ones_like(xyz[:, 0]), xyz_pred_sc_tr_aligned)

                # align predictions
                xyz_pred_aligned = self.align_w_scale(xyz, xyz_pred)
                verts_pred_aligned = self.align_w_scale(verts, verts_pred)

                self.eval_xyz_procrustes_aligned.feed(xyz, np.ones_like(xyz[:, 0]), xyz_pred_aligned)
                self.eval_mesh_err_aligned.feed(verts, np.ones_like(verts[:, 0]), verts_pred_aligned)

                # F-scores
                l, la = list(), list()
                for t in self.f_threshs:
                    # for each threshold calculate the f score and the f score of the aligned vertices
                    f, _, _ = self.calculate_fscore(verts, verts_pred, t)
                    # f = 0.
                    l.append(f)
                    f, _, _ = self.calculate_fscore(verts, verts_pred_aligned, t)
                    # f = 0.
                    la.append(f)
                self.f_score.append(l)
                self.f_score_aligned.append(la)

                self.n_eval_samples += 1

    def on_finished(self):
        # Calculate results
        print('Total number of samples: %d' % self.n_eval_samples)

        xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = self.eval_xyz.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D KP results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

        xyz_procrustes_al_mean3d, _, xyz_procrustes_al_auc3d, pck_xyz_procrustes_al, thresh_xyz_procrustes_al = \
            self.eval_xyz_procrustes_aligned.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D KP PROCRUSTES ALIGNED results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_procrustes_al_auc3d, xyz_procrustes_al_mean3d * 100.0))

        xyz_sc_tr_al_mean3d, _, xyz_sc_tr_al_auc3d, pck_xyz_sc_tr_al, thresh_xyz_sc_tr_al = \
            self.eval_xyz_sc_tr_aligned.get_measures(
            0.0, 0.05, 100)
        print('Evaluation 3D KP SCALE-TRANSLATION ALIGNED results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (xyz_sc_tr_al_auc3d, xyz_sc_tr_al_mean3d * 100.0))

        mesh_mean3d, _, mesh_auc3d, pck_mesh, thresh_mesh = self.eval_mesh_err.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (mesh_auc3d, mesh_mean3d * 100.0))

        mesh_al_mean3d, _, mesh_al_auc3d, pck_mesh_al, thresh_mesh_al = \
            self.eval_mesh_err_aligned.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH ALIGNED results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (mesh_al_auc3d, mesh_al_mean3d * 100.0))

        print('F-scores')
        f_out = list()
        f_score, f_score_aligned = np.array(self.f_score).T, np.array(self.f_score_aligned).T
        for f, fa, t in zip(f_score, f_score_aligned, self.f_threshs):
            print('F@%.1fmm = %.3f' % (t * 1000, f.mean()), '\tF_aligned@%.1fmm = %.3f' % (t * 1000, fa.mean()))
            f_out.append('f_score_%d: %f' % (round(t * 1000), f.mean()))
            f_out.append('f_al_score_%d: %f' % (round(t * 1000), fa.mean()))

        score_path = os.path.join(self.preds_dir, 'scores.txt')

        with open(score_path, 'w') as fo:
            xyz_mean3d *= 100
            xyz_procrustes_al_mean3d *= 100
            xyz_sc_tr_al_mean3d *= 100
            fo.write('xyz_mean3d: %f\n' % xyz_mean3d)
            fo.write('xyz_auc3d: %f\n' % xyz_auc3d)
            fo.write('xyz_procrustes_al_mean3d: %f\n' % xyz_procrustes_al_mean3d)
            fo.write('xyz_procrustes_al_auc3d: %f\n' % xyz_procrustes_al_auc3d)
            fo.write('xyz_scale_trans_al_mean3d: %f\n' % xyz_sc_tr_al_mean3d)
            fo.write('xyz_scale_trans_al_auc3d: %f\n' % xyz_sc_tr_al_auc3d)

            mesh_mean3d *= 100
            mesh_al_mean3d *= 100
            fo.write('mesh_mean3d: %f\n' % mesh_mean3d)
            fo.write('mesh_auc3d: %f\n' % mesh_auc3d)
            fo.write('mesh_al_mean3d: %f\n' % mesh_al_mean3d)
            fo.write('mesh_al_auc3d: %f\n' % mesh_al_auc3d)
            for t in f_out:
                fo.write('%s\n' % t)
        print('Scores written to: %s' % score_path)


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
        batch_size = len(cam_num)

        gt_vts = inputs["master_verts_3d"].reshape(batch_size, 778, 3)
        gt_jts = inputs["master_joints_3d"].reshape(batch_size, 21, 3)

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
