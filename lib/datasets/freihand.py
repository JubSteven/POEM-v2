import hashlib
import json
import os
import pickle
import shutil
import subprocess
from typing import List

import imageio
import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from termcolor import colored

from ..utils.builder import DATASET
from ..utils.logger import logger
from ..utils.transform import (bbox_xywh_to_xyxy, get_annot_center, get_annot_scale, persp_project)
from .hdata import HDataset, kpId2vertices


def split_mano_param(mano_param):
    poses = mano_param[:, :48]
    shapes = mano_param[:, 48:58]
    uv_root = mano_param[:, 58:60]
    scale = mano_param[:, 60:]
    return poses, shapes, uv_root, scale


def get_focal_pp(K):
    """Extract the camera parameters that are relevant for an orthographic assumption."""
    focal = 0.5 * (K[0, 0] + K[1, 1])
    pp = K[:2, 2]
    return focal, pp


def backproject_ortho(uv, scale, focal, pp):  # kind of the predictions  # kind of the camera calibration
    """Calculate 3D coordinates from 2D coordinates and the camera parameters."""
    uv = uv.copy()
    uv -= pp
    xyz = np.concatenate([np.reshape(uv, [-1, 2]), np.ones_like(uv[:, :1]) * focal], 1)
    xyz /= scale
    return xyz


def recover_root(uv_root, scale, focal, pp):
    uv_root = np.reshape(uv_root, [1, 2])
    xyz_root = backproject_ortho(uv_root, scale, focal, pp)
    return xyz_root


def get_keypoints_from_mesh_th(mesh_vertices, keypoints_regressed):
    """Assembles the full 21 keypoint set from the 16 Mano Keypoints and 5 mesh vertices for the fingers."""
    keypoints = [0.0 for _ in range(21)]  # init empty list

    # fill keypoints which are regressed
    mapping = {
        0: 0,  # Wrist
        1: 5,
        2: 6,
        3: 7,  # Index
        4: 9,
        5: 10,
        6: 11,  # Middle
        7: 17,
        8: 18,
        9: 19,  # Pinky
        10: 13,
        11: 14,
        12: 15,  # Ring
        13: 1,
        14: 2,
        15: 3,  # Thumb
    }

    for manoId, myId in mapping.items():
        keypoints[myId] = keypoints_regressed[manoId, :]

    # get other keypoints from mesh
    for myId, meshId in kpId2vertices.items():
        keypoints[myId] = torch.mean(mesh_vertices[meshId, :], 0)

    keypoints = torch.vstack(keypoints)

    return keypoints


def freihand_param_conversion(mano_param, layer, cam_intr, anno_center_idx=9):
    pose, shape, uv_root, scale = split_mano_param(mano_param)

    pose_th = torch.FloatTensor(pose)
    shape_th = torch.FloatTensor(shape)
    outcome = layer(pose_th, shape_th)
    mesh_xyz = outcome.verts
    pose_vec_xyz = outcome.full_poses
    mesh_xyz = mesh_xyz.squeeze(0)
    pose_vec_xyz = pose_vec_xyz.squeeze(0)

    focal, pp = get_focal_pp(cam_intr)
    xyz_root = recover_root(uv_root, scale, focal, pp)
    xyz_root = torch.tensor(xyz_root)

    pose_xyz_regressed = torch.matmul(layer.th_J_regressor, mesh_xyz)
    pose_xyz = get_keypoints_from_mesh_th(mesh_xyz, pose_xyz_regressed)

    global_tsl = xyz_root - pose_xyz[anno_center_idx]

    mesh_xyz = mesh_xyz + global_tsl
    pose_xyz = pose_xyz + global_tsl
    return mesh_xyz, pose_xyz, pose_vec_xyz, shape_th.squeeze(0)


def freihand_dump_json(pred_out_path, xyz_pred_list, verts_pred_list, codalab=True):
    """Dump joints and verts prediction of FreiHAND test into .json and .zip
    for Codalab online evaluation.

    Args:
        pred_out_path (str): path to dump file
        xyz_pred_list (list[JOINTS]): list of predicted joint
        verts_pred_list (list[VERTS]): list of predicted verts
        codalab (bool, optional): whether submitting to codalab. Defaults to True.

    Returns:
        None
    """  # make sure its only lists

    def roundall(rows):
        return [[round(val, 5) for val in row] for row in rows]

    xyz_pred_list = [roundall(x.tolist()) for x in xyz_pred_list]
    verts_pred_list = [roundall(x.tolist()) for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, "w") as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)
    print("Dumped %d joints and %d verts predictions to %s" % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))
    if codalab:
        file_name = ".".join(pred_out_path.split("/")[-1].split(".")[:-1])
        if pred_out_path != f"./common/{file_name}.json":
            shutil.copy(pred_out_path, f"./common/{file_name}.json")
        subprocess.call(["zip", "-j", f"./common/{file_name}.zip", f"./common/{file_name}.json"])


@DATASET.register_module()
class FreiHAND(HDataset):
    """Mano version and FreiHAND v2
    The following paragraph is only relevant if you downloaded the dataset before 22.
    reference: https://github.com/lmb-freiburg/freihand
    """

    TRAINING_UNIQUE_SIZE = 32560
    TRAINING_ALL_SIZE = 130240
    EVALUATION_SIZE = 3960
    IMG_WIDTH = 224
    IMG_HEIGHT = 224

    def __init__(self, cfg):
        super().__init__(cfg)
        self.split_mode = cfg.SPLIT_MODE
        self.mode_opts = ["full", "unique"]
        self._mano_layer = ManoLayer(joint_rot_mode="axisang",
                                     use_pca=False,
                                     mano_assets_root="assets/mano_v1_2",
                                     center_idx=None,
                                     flat_hand_mean=False)
        self.mano_mean_pose = self._mano_layer.th_hands_mean.numpy()  # (1, 45)

        if self.data_split == "test":
            self.use_pseudo_joints = cfg.USE_PSEUDO_JOINTS
            self.use_pseudo_bbox = cfg.USE_PSEUDO_BBOX
            self.split_mode = "full"  # eval set always full mode

        self.load_dataset()
        logger.info(f"initialized child class: {self.name} (HDataset)")

    def _preload(self):
        self.name = "FreiHAND_v2"
        self.root = os.path.join(self.data_root, self.name)
        self.root_supp = os.path.join(self.data_root, f"{self.name}_supp")
        self.root_extra_info = os.path.join(os.path.normpath("assets"), self.name)

        if self.data_split == "train":
            self.data_split_subfolder = "training"
            if self.split_mode == "full":
                self.n_samples = FreiHAND.TRAINING_ALL_SIZE
            elif self.split_mode == "unique":
                self.n_samples = FreiHAND.TRAINING_UNIQUE_SIZE
            else:
                raise ValueError(f"split_mode in [train] not followed freihand mode options [full|unique]")
        elif self.data_split == "test":
            self.data_split_subfolder = "evaluation"
            self.n_samples = FreiHAND.EVALUATION_SIZE
        else:
            raise ValueError(f"data_split {self.data_split} not followed FreiHandV2 partition [train|test]")

        self.cache_identifier_dict = {
            "data_split": self.data_split,
            "split_mode": self.split_mode,
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format(self.cache_identifier))

    def load_dataset(self):
        self._preload()
        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        if os.path.exists(self.cache_path) and self.use_cache:  # load from cache
            with open(self.cache_path, "rb") as p_f:
                annotations = pickle.load(p_f)
            logger.info(f"Loaded cache for {self.name}_{self.data_split}_{self.split_mode} from {self.cache_path}")
            self.image_paths = annotations["image_paths"]
            self.mask_paths = annotations["mask_paths"]
            self.cam_intrs = annotations["cam_intrs"]
            self.bone_scale = annotations["bone_scale"]
            self.joints_3d = annotations["joints_3d"]
            self.verts_3d = annotations["verts_3d"]
            self.raw_mano_param = annotations["raw_mano_param"]
        else:  # load from disk
            logger.warning(f"Loading {self.name}_{self.data_split}_{self.split_mode} from disk, take a while ...")
            self.image_paths = []
            self.mask_paths = []
            for i in range(self.n_samples):
                self.image_paths.append(os.path.join(self.root, self.data_split_subfolder, "rgb", "%08d.jpg" % i))
                if self.data_split == "train":
                    i = i % self.TRAINING_UNIQUE_SIZE
                    self.mask_paths.append(os.path.join(self.root, self.data_split_subfolder, "mask", "%08d.jpg" % i))

            raw_intr = json.loads(open(os.path.join(self.root, self.data_split_subfolder + "_K.json")).read())
            raw_scale = json.loads(open(os.path.join(self.root, self.data_split_subfolder + "_scale.json")).read())
            self.cam_intrs = np.array(raw_intr, dtype=np.float32)
            self.bone_scale = np.array(raw_scale, dtype=np.float32)

            if self.data_split == "train":
                raw_joints = json.loads(open(os.path.join(self.root, self.data_split_subfolder + "_xyz.json")).read())
                raw_verts = json.loads(open(os.path.join(self.root, self.data_split_subfolder + "_verts.json")).read())
                raw_mano = json.loads(open(os.path.join(self.root, self.data_split_subfolder + "_mano.json")).read())
                self.joints_3d = np.array(raw_joints, dtype=np.float32)
                self.verts_3d = np.array(raw_verts, dtype=np.float32)
                self.raw_mano_param = np.array(raw_mano, dtype=np.float32)
            else:
                self.joints_3d = np.array([])
                self.verts_3d = np.array([])
                self.raw_mano_param = np.array([])

            annotations = {
                "cache_identifier_dict": self.cache_identifier_dict,
                "image_paths": self.image_paths,
                "mask_paths": self.mask_paths,
                "cam_intrs": self.cam_intrs,
                "bone_scale": self.bone_scale,
                "joints_3d": self.joints_3d,
                "verts_3d": self.verts_3d,
                "raw_mano_param": self.raw_mano_param,
            }

            # if self.use_cache:  # dump cache
            with open(self.cache_path, "wb") as fid:
                pickle.dump(annotations, fid)
            logger.info(f"Wrote cache: {self.name} {self.data_split} {self.split_mode} to {self.cache_path}")

        if self.data_split == "test":
            if self.use_pseudo_bbox:
                # * add I2L-MeshNet bbox:
                self.test_bbox = []
                logger.warning("load in I2L-MeshNet prediction bbox")
                pred_res = json.load(open(os.path.join(self.root_extra_info, "freihand_eval_coco.json"), "r"))
                pred_res = pred_res["annotations"]
                assert len(pred_res) == self.n_samples, "wrong evaluation set size"
                for elm in pred_res:
                    pred_bbox = np.array(elm["bbox"])[None, :]  # xmin, ymin, width, height
                    self.test_bbox.append(pred_bbox)

                self.test_bbox = np.concatenate(self.test_bbox)  # (B, 4)
                assert self.test_bbox.shape == (self.n_samples, 4), "wrong bbox annotation size"

            # * add Xinyu Chen's joints predictions
            if self.use_pseudo_joints:
                logger.warning("load in CMR-SG prediction joints")
                pred_res = json.load(open(os.path.join(self.root_extra_info, "cmr_sg.json"), "r"))
                pred_joints = np.array(pred_res[0], dtype=np.float32)
                self.joints_3d = pred_joints
                assert self.joints_3d.shape == (self.n_samples, 21, 3), "wrong joints annotation size"

        self.sample_idxs = list(range(self.n_samples))
        logger.info(f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])}"
                    f"/{self.n_samples} samples for data_split {self.data_split}")

        return True

    def __len__(self):
        return len(self.sample_idxs)

    def get_sample_idx(self) -> List[int]:
        return self.sample_idxs

    def get_image(self, idx):
        path = self.image_paths[idx]
        image = np.array(imageio.imread(path, pilmode="RGB"), dtype=np.uint8)
        return image

    def get_rawimage_size(self, idx):
        return [224, 224]

    def get_image_mask(self, idx):
        if len(self.mask_paths) != 0:
            path = self.mask_paths[idx]
            mask = np.array(imageio.imread(path, as_gray=True), dtype=np.uint8)
        else:
            mask = np.zeros((FreiHAND.IMG_HEIGHT, FreiHAND.IMG_WIDTH), dtype=np.uint8)
        return mask

    def get_image_path(self, idx):
        return self.image_paths[idx]

    def get_cam_intr(self, idx):
        if self.data_split == "train":
            idx = idx % self.TRAINING_UNIQUE_SIZE
        intr = self.cam_intrs[idx].copy()
        return intr

    def get_cam_center(self, idx):
        intr = self.get_cam_intr(idx)
        return np.array([intr[0, 2], intr[1, 2]])

    def get_hand_faces(self, idx):
        return self._mano_layer.get_mano_closed_faces().numpy()

    def get_joints_3d(self, idx):
        if self.data_split == "train":
            idx = idx % self.TRAINING_UNIQUE_SIZE
        if self.data_split == "train" or self.use_pseudo_joints:
            joints_3d = self.joints_3d[idx].copy()
            return joints_3d
        else:
            return np.zeros((21, 3), dtype=np.float32)

    def get_verts_3d(self, idx):
        if self.data_split == "train":
            idx = idx % self.TRAINING_UNIQUE_SIZE

        if self.data_split != "test":
            verts_3d = self.verts_3d[idx]
            return verts_3d.copy()
        else:  # dummy
            return np.zeros((778, 3), dtype=np.float32)

    def get_joints_2d(self, idx):
        if self.data_split == "train" or self.use_pseudo_joints:
            joints_3d = self.get_joints_3d(idx)
            intr = self.get_cam_intr(idx)
            return persp_project(joints_3d, intr)
        else:
            return np.zeros((21, 2), dtype=np.float32)

    def get_joints_uvd(self, idx):
        uv = self.get_joints_2d(idx)
        d = self.get_joints_3d(idx)[:, 2:]  # (21, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_verts_uvd(self, idx):
        v3d = self.get_verts_3d(idx)
        intr = self.get_cam_intr(idx)
        uv = persp_project(v3d, intr)[:, :2]
        d = v3d[:, 2:]  # (778, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_verts_2d(self, idx):
        if self.data_split != "test":
            v3d = self.get_verts_3d(idx)
            intr = self.get_cam_intr(idx)
            return persp_project(v3d, intr)
        else:
            return np.zeros((778, 2), dtype=np.float32)

    def get_sides(self, idx):
        return "right"

    def get_bone_scale(self, idx):
        if self.data_split == "train":
            idx = idx % self.TRAINING_UNIQUE_SIZE

        return self.bone_scale[idx]

    def get_bbox_center_scale(self, idx):
        if self.data_split == "train":
            joints_2d = self.get_joints_2d(idx)
            center = get_annot_center(joints_2d)
            scale = get_annot_scale(joints_2d)
            return center, scale
        elif self.use_pseudo_bbox:
            xywh = list(self.test_bbox[idx])
            xyxy = bbox_xywh_to_xyxy(xywh)
            c_x = int((xyxy[0] + xyxy[2]) / 2)
            c_y = int((xyxy[1] + xyxy[3]) / 2)
            center = np.asarray([c_x, c_y])
            scale = max(xywh[2], xywh[3])
            return center, scale
        else:
            center = np.asarray([112, 112], dtype=np.float32)
            scale = 224 * 0.50
            return center, scale

    def get_raw_mano_param(self, idx):
        if self.data_split == "train":
            idx = idx % self.TRAINING_UNIQUE_SIZE

        return self.raw_mano_param[idx]

    def get_mano_pose(self, idx):
        if self.data_split == "train":
            idx = idx % self.TRAINING_UNIQUE_SIZE
            raw_pose = split_mano_param(self.raw_mano_param[idx])[0].copy()  # (1, 48)
            raw_pose_rel = raw_pose[:, 3:]  # (1, 45)
            raw_pose_wrist = raw_pose[:, :3]  # (1, 3)
            new_pose = np.concatenate([raw_pose_wrist, self.mano_mean_pose + raw_pose_rel], 1)
            return new_pose.squeeze(0)  # (48)
        else:
            return np.zeros((48), dtype=np.float32)

    def get_mano_shape(self, idx):
        if self.data_split == "train":
            idx = idx % self.TRAINING_UNIQUE_SIZE
            raw_beta = split_mano_param(self.raw_mano_param[idx])[1].copy()  # (1, 10)
            return raw_beta.squeeze(0)
        else:
            return np.zeros((10), dtype=np.float32)

    def get_sample_identifier(self, idx):
        res = f"{self.name}__{self.cache_identifier_raw}__{idx}"
        return res


@DATASET.register_module()
class FreiHAND_v2_Extra(FreiHAND):

    def __init__(self, cfg):
        if cfg.DATA_SPLIT == "train":
            # Initializing FreiHAND_v2_Extra training set will call FreiHAND.__init__
            super(FreiHAND_v2_Extra, self).__init__(cfg)  # FreiHAND's Initialization
        else:
            super(FreiHAND, self).__init__(cfg)  # HData's Initialization
            self.split_mode = cfg.SPLIT_MODE
            self.mode_opts = ["full", "unique"]
            assert self.split_mode == "full", "val and test only support split_mode: full"
            self._mano_layer = ManoLayer(joint_rot_mode="axisang",
                                         use_pca=False,
                                         mano_assets_root="assets/mano_v1_2",
                                         center_idx=None,
                                         flat_hand_mean=False)
            self.mano_mean_pose = self._mano_layer.th_hands_mean.numpy()  # (1, 45)
            self.load_dataset_extra()

        logger.warning(f"Initialized child class: FreiHAND_v2_Extra (FreiHAND)")

    def _preload_extra(self):
        self.name = "FreiHAND_v2_Extra"
        self.root = os.path.join(self.data_root, f"FreiHAND_v2_eval")  # data/FreiHAND_v2_eval
        self.root_supp = os.path.join(self.data_root, f"FreiHAND_v2_supp")

        if self.data_split == "val":
            self.data_split_subfolder = "evaluation"
            self.n_samples = FreiHAND.EVALUATION_SIZE
        elif self.data_split == "test":
            self.data_split_subfolder = "evaluation"
            self.n_samples = FreiHAND.EVALUATION_SIZE
        else:
            raise ValueError(f"data_split {self.data_split} not followed FreiHand_v2_Extra partition [val|test]")

        self.cache_identifier_dict = {
            "dataset_name": self.name,
            "data_split": self.data_split,
            "split_mode": self.split_mode,
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format(self.cache_identifier))

    def load_dataset_extra(self):
        self._preload_extra()
        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        if os.path.exists(self.cache_path) and self.use_cache:  # load from cache
            with open(self.cache_path, "rb") as p_f:
                annotations = pickle.load(p_f)
            logger.info(f"Loaded cache for {self.name}_{self.data_split}_{self.split_mode} from {self.cache_path}")
            self.image_paths = annotations["image_paths"]
            self.mask_paths = annotations["mask_paths"]
            self.cam_intrs = annotations["cam_intrs"]
            self.bone_scale = annotations["bone_scale"]
            self.joints_3d = annotations["joints_3d"]
            self.verts_3d = annotations["verts_3d"]
            self.raw_mano_param = annotations["raw_mano_param"]
        else:
            logger.warning(f"Loading {self.name} {self.data_split} from disk, take a while ...")
            self.image_paths = []
            self.mask_paths = []
            for i in range(self.n_samples):
                self.image_paths.append(os.path.join(self.root, self.data_split_subfolder, "rgb", "%08d.jpg" % i))
                self.mask_paths.append(os.path.join(self.root, self.data_split_subfolder, "segmap", "%08d.png" % i))

            raw_intr = json.loads(open(os.path.join(self.root, self.data_split_subfolder + "_K.json")).read())
            raw_scale = json.loads(open(os.path.join(self.root, self.data_split_subfolder + "_scale.json")).read())
            raw_joints = json.loads(open(os.path.join(self.root, self.data_split_subfolder + "_xyz.json")).read())
            raw_verts = json.loads(open(os.path.join(self.root, self.data_split_subfolder + "_verts.json")).read())
            raw_mano = json.loads(open(os.path.join(self.root, self.data_split_subfolder + "_mano.json")).read())

            self.cam_intrs = np.array(raw_intr, dtype=np.float32)
            self.bone_scale = np.array(raw_scale, dtype=np.float32)
            self.joints_3d = np.array(raw_joints, dtype=np.float32)
            self.verts_3d = np.array(raw_verts, dtype=np.float32)
            self.raw_mano_param = np.array(raw_mano, dtype=np.float32)

            annotations = {
                "cache_identifier_dict": self.cache_identifier_dict,
                "image_paths": self.image_paths,
                "mask_paths": self.mask_paths,
                "cam_intrs": self.cam_intrs,
                "bone_scale": self.bone_scale,
                "joints_3d": self.joints_3d,
                "verts_3d": self.verts_3d,
                "raw_mano_param": self.raw_mano_param,
            }

            if self.use_cache:  # dump cache
                with open(self.cache_path, "wb") as fid:
                    pickle.dump(annotations, fid)
                logger.info(f"Wrote cache: {self.name} {self.data_split} {self.split_mode} to {self.cache_path}")

        self.sample_idxs = list(range(self.n_samples))
        logger.info(f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])}"
                    f"/{self.n_samples} samples for data_split {self.data_split}")

    def get_image_mask(self, idx):
        if self.data_split == "train":
            return super().get_image_mask(idx)

        path = self.mask_paths[idx]
        mask = np.array(imageio.imread(path, as_gray=True), dtype=np.uint8)
        mask[mask >= 1] = 255
        return mask

    def get_joints_3d(self, idx):
        if self.data_split == "train":
            return super().get_joints_3d(idx)

        return self.joints_3d[idx].copy()

    def get_verts_3d(self, idx):
        if self.data_split == "train":
            return super().get_verts_3d(idx)

        return self.verts_3d[idx].copy()

    def get_joints_2d(self, idx):
        joints_3d = self.get_joints_3d(idx)
        intr = self.get_cam_intr(idx)
        return persp_project(joints_3d, intr)

    def get_verts_2d(self, idx):
        verts_3d = self.get_verts_3d(idx)
        intr = self.get_cam_intr(idx)
        return persp_project(verts_3d, intr)

    def get_bbox_center_scale(self, idx):
        joints_2d = self.get_joints_2d(idx)
        center = get_annot_center(joints_2d)
        scale = get_annot_scale(joints_2d)
        return center, scale

    def get_raw_mano_param(self, idx):
        if self.data_split == "train":
            return super().get_raw_mano_param(idx)

        return self.raw_mano_param[idx].copy()

    def get_mano_pose(self, idx):
        if self.data_split == "train":
            return super().get_mano_pose(idx)

        raw_pose = split_mano_param(self.raw_mano_param[idx])[0].copy()  # (1, 48)
        raw_pose_rel = raw_pose[:, 3:]  # (1, 45)
        raw_pose_wrist = raw_pose[:, :3]  # (1, 3)
        new_pose = np.concatenate([raw_pose_wrist, self.mano_mean_pose + raw_pose_rel], 1)
        return new_pose.squeeze(0)  # (48)

    def get_mano_shape(self, idx):
        if self.data_split == "train":
            return super().get_mano_shape(idx)

        raw_beta = split_mano_param(self.raw_mano_param[idx])[1].copy()  # (1, 10)
        return raw_beta.squeeze(0)
