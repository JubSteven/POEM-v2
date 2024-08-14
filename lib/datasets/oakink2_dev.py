# @TODO: xinyu
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import List
import warnings
import cv2

import imageio
import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from termcolor import colored
import random
from functools import lru_cache

from ..utils.builder import DATASET
from ..utils.etqdm import etqdm
from ..utils.logger import logger
from ..utils.transform import get_annot_center, get_annot_scale, persp_project
from ..utils.transform import quat_to_rotmat, rotmat_to_aa, quat_to_aa
from .hdata import HDataset, kpId2vertices


@DATASET.register_module()
class OakInk2_Dev(HDataset):
    def __init__(self, cfg) -> None:
        super(OakInk2_Dev, self).__init__(cfg)
        self.rMANO = ManoLayer(side="right", mano_assets_root="assets/mano_v1_2")
        self.lMANO = ManoLayer(side="left", mano_assets_root="assets/mano_v1_2")
        assert self.data_mode in [
            "2D",
            "UVD",
            "3D",
        ], f"OakInk2_Dev does not dupport {self.data_mode} mode"
        assert self.data_split in [
            "all",
            "train+val",
            "train",
            "val",
            "test",
        ], "OakInk2_Dev data_split must be one of ['all', 'train+val', 'train', 'val', 'test']"
        self.split_mode = cfg.SPLIT_MODE
        self.use_mv = cfg.get(
            "USE_MV", False
        )  # This option affects which dir it will look up data samples and metas
        assert self.split_mode in [
            "default",
        ], "OakInk split_mode must be one of ['default']"

        self.load_dataset()
        logger.info(f"initialized child class: {self.name}")
        if self.use_mv:
            logger.info(f"    {self.name}: use_mv")

    def load_dataset(self):
        self.name = self.__class__.__name__

        self.image_root = os.path.join(self.data_root, "OakInk2_Dev")
        if not self.use_mv:
            self.annot_root = os.path.join(self.data_root, "OakInk2_Dev", "packed_anno")
            split_file = os.path.join(self.annot_root, "split_meta.json")
        else:
            self.annot_root = os.path.join(
                self.data_root, "OakInk2_Dev", "packed_anno_mv"
            )
            split_file = os.path.join(self.annot_root, "split_meta.json")
        with open(split_file, "r") as f:
            split_meta = json.load(f)
        if self.data_split == "all":
            split_tuple_list = (
                split_meta["train"] + split_meta["val"] + split_meta["test"]
            )
        elif self.data_split == "train+val":
            split_tuple_list = split_meta["train"] + split_meta["val"]
        else:
            split_tuple_list = split_meta[self.data_split]
        # self.split_tuple_list = [el for el in split_tuple_list if el[4] == "lh"]
        # self.split_tuple_list = split_tuple_list[:int(len(split_tuple_list)*0.01)]
        self.split_tuple_list = split_tuple_list

        logger.info(
            f"{self.name} Got {colored(len(self.split_tuple_list), 'yellow', attrs=['bold'])}"
            f"/{len(split_tuple_list)} samples for data_split {self.data_split}"
        )

    def __len__(self):
        return len(self.split_tuple_list)

    @lru_cache(maxsize=1)
    def _load_pickle(self, idx):
        split_tuple = self.split_tuple_list[idx]
        split, pk, f_id, cam_serial, hand_side = split_tuple
        anno_info_filepath = os.path.join(
            self.annot_root,
            split,
            pk,
            f"anno_{hand_side}",
            cam_serial,
            f"{f_id:0>6}.pkl",
        )
        with open(anno_info_filepath, "rb") as f:
            data = pickle.load(f)
        return data

    def get_image_path(self, idx):
        anno_info = self._load_pickle(idx)
        image_path = anno_info["image_path"]
        image_path = os.path.join(self.image_root, image_path)
        return image_path

    def get_image(self, idx):
        path = self.get_image_path(idx)
        image = np.array(imageio.imread(path, pilmode="RGB"), dtype=np.uint8)
        return image

    def get_rawimage_size(self, idx):
        # MUST (W, H)
        return (848, 480)

    def get_image_mask(self, idx):
        # mask_path = os.path.join(self.root, "mask", f"{self.info_str_list[idx]}.png")
        # mask = np.array(imageio.imread(mask_path, as_gray=True), dtype=np.uint8)
        # return mask
        return np.zeros((480, 848), dtype=np.uint8)

    def get_cam_intr(self, idx):
        anno_info = self._load_pickle(idx)
        cam_intr = anno_info["cam_intr"]
        return cam_intr.astype(np.float32)

    def get_cam_center(self, idx):
        intr = self.get_cam_intr(idx)
        return np.array([intr[0, 2], intr[1, 2]])

    def get_hand_faces(self, idx):
        return self.rMANO.get_mano_closed_faces().numpy()

    def get_joints_3d(self, idx):
        anno_info = self._load_pickle(idx)
        joints_3d = anno_info["joints_cam_rgrd"]
        return joints_3d.astype(np.float32)

    def get_verts_3d(self, idx):
        anno_info = self._load_pickle(idx)
        verts_3d = anno_info["verts_cam"]
        return verts_3d.astype(np.float32)

    def get_joints_2d(self, idx):
        anno_info = self._load_pickle(idx)
        joints_2d = anno_info["joints_2d_rgrd"]
        return joints_2d.astype(np.float32)

    def get_joints_uvd(self, idx):
        uv = self.get_joints_2d(idx)
        d = self.get_joints_3d(idx)[:, 2:]  # (21, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_verts_2d(self, idx):
        anno_info = self._load_pickle(idx)
        verts_2d = anno_info["verts_2d"]
        return verts_2d.astype(np.float32)

    def get_verts_uvd(self, idx):
        uv = self.get_verts_2d(idx)
        d = self.get_verts_3d(idx)[:, 2:]  # (21, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_sides(self, idx):
        split_tuple = self.split_tuple_list[idx]
        split, pk, f_id, cam_serial, hand_side = split_tuple
        if hand_side == "lh":
            return "left"
        else:
            return "right"

    def get_bone_scale(self, idx):
        raise NotImplementedError(f"{self.name} does not support bone scale")

    def get_bbox_center_scale(self, idx):
        joints_2d = self.get_joints_2d(idx)
        center = get_annot_center(joints_2d)
        scale = get_annot_scale(joints_2d)
        return center, scale

    def get_mano_pose(self, idx):
        anno_info = self._load_pickle(idx)
        mano_pose = anno_info["mano_pose_cam"]
        mano_pose_np = mano_pose.numpy()
        return mano_pose_np.astype(np.float32)

    def get_mano_shape(self, idx):
        anno_info = self._load_pickle(idx)
        mano_shape = anno_info["mano_shape"]
        mano_shape_np = mano_shape.numpy()
        return mano_shape_np.astype(np.float32)

    def get_sample_identifier(self, idx):
        res = f"{self.name}__{self.split_tuple_list[idx]}"
        return res

    def getitem_3d(self, idx):
        split_tuple = self.split_tuple_list[idx]
        split, pk, f_id, cam_serial, hand_side = split_tuple
        if hand_side == "lh":
            hand_side = "left"
        else:
            hand_side = "right"
        # load pack
        anno_info = self._load_pickle(idx)

        cam_intr = np.array(anno_info["cam_intr"], dtype=np.float32)
        joints_3d = np.array(anno_info["joints_cam_rgrd"], dtype=np.float32)
        verts_3d = np.array(anno_info["verts_cam"], dtype=np.float32)
        joints_2d = np.array(anno_info["joints_2d_rgrd"], dtype=np.float32)
        verts_2d = np.array(anno_info["verts_2d"], dtype=np.float32)
        joints_uvd = np.concatenate((joints_2d, joints_3d[:, 2:]), axis=1)
        verts_uvd = np.concatenate((verts_2d, verts_3d[:, 2:]), axis=1)

        bbox_center = get_annot_center(joints_2d)
        bbox_scale = get_annot_scale(joints_2d)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle
        cam_center = np.array([cam_intr[0, 2], cam_intr[1, 2]])

        image_path = self.get_image_path(idx)
        image = self.get_image(idx)
        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = self.get_joints_2d_vis(joints_2d=joints_2d, raw_size=raw_size)

        image_mask = self.get_image_mask(idx)

        flip_hand = True if hand_side != self.sides else False

        # for mv dataset
        cam_extr = np.array(anno_info["cam_extr"])

        # Flip 2d if needed
        if flip_hand:
            bbox_center[0] = 2 * cam_center[0] - bbox_center[0]  # image center
            joints_3d = self.flip_3d(joints_3d)
            verts_3d = self.flip_3d(verts_3d)
            joints_uvd = self.flip_2d(joints_uvd, 2 * cam_center[0])
            verts_uvd = self.flip_2d(verts_uvd, 2 * cam_center[0])
            joints_2d = self.flip_2d(joints_2d, 2 * cam_center[0])
            # image & mask should be flipped horizontally with center at cam_center[0]
            # use cv2
            M = np.array([[-1, 0, 2 * cam_center[0]], [0, 1, 0]], dtype=np.float32)
            # Use warpAffine to apply the reflection
            image = cv2.warpAffine(image, M, raw_size)
            image_mask = cv2.warpAffine(image_mask, M, raw_size)

        label = {
            "idx": idx,
            "cam_center": cam_center,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "cam_intr": cam_intr,
            "joints_2d": joints_2d,
            "joints_3d": joints_3d,
            "verts_3d": verts_3d,
            "joints_vis": joints_vis,
            "joints_uvd": joints_uvd,
            "verts_uvd": verts_uvd,
            "image_path": image_path,
            "raw_size": raw_size,
            "image_mask": image_mask,
            "cam_extr": cam_extr,
            "cam_serial": cam_serial,
            # "_image": image,
        }
        if self.use_mv:
            label["hand_side"] = hand_side

        results = self.transform(image, label)
        results.update(label)
        return results


@DATASET.register_module()
class OakInk2_Dev_MultiView(torch.utils.data.Dataset):
    def __init__(self, cfg) -> None:
        super(OakInk2_Dev_MultiView, self).__init__()
        self.name = self.__class__.__name__

        # data settings
        # these code present here for narrowing check
        # as multiview dataset propose a more strict constraint over config
        self.data_mode = cfg.DATA_MODE
        self.use_quarter = cfg.get("USE_QUARTER", False)
        self.skip_frames = cfg.get("SKIP_FRAMES", 0)
        if self.use_quarter is True:
            warnings.warn(f"use_quarter is deprecated, use skip_frames=3 instead")
            self.skip_frames = 3

        assert self.data_mode == "3D", f"{self.name} only support 3D data mode"
        self.data_split = cfg.DATA_SPLIT
        assert self.data_split in [
            "all",
            "train+val",
            "train",
            "val",
            "test",
        ], "OakInk2_Dev data_split must be one of <see the src>"
        self.split_mode = cfg.SPLIT_MODE
        assert self.split_mode in [
            "default"
        ], "OakInk2_Dev split_mode must be one of ['subject', 'object]"

        # multiview settings
        self.n_views = cfg.N_VIEWS
        assert self.n_views == 4, f"{self.name} only support 4 view"

        # For generation to random view num
        self.random_n_views = (
            cfg.RANDOM_N_VIEWS
        )  # whether to truncate the batch into size min_views to N_VIEWS
        self.min_views = (
            0 if not self.random_n_views else cfg.MIN_VIEWS
        )  # minimum views required

        self.test_with_multiview = cfg.get("TEST_WITH_MULTIVIEW", False)
        self.master_system = cfg.get("MASTER_SYSTEM", "as_constant_camera")
        assert self.master_system in [
            "as_constant_camera",
        ], f"{self.name} got unsupport master system mode {self.master_system}"

        self.const_cam_view_id = 0

        # preset
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX

        # load dataset (single view for given split)
        self._dataset = OakInk2_Dev(cfg)

        # view info
        view_info_filepath = os.path.join(self._dataset.annot_root, "split_view.json")
        with open(view_info_filepath, "r") as f:
            view_info = json.load(f)
        if self.data_split == "all":
            view_idx_list = view_info["train"] + view_info["val"] + view_info["test"]
        elif self.data_split == "train+val":
            view_idx_list = view_info["train"] + view_info["val"]
        else:
            view_idx_list = view_info[self.data_split]
        self._mv_list = view_idx_list

        if self.skip_frames != 0:
            self.valid_sample_idx_list = [
                i for i in range(len(self._mv_list)) if i % (self.skip_frames + 1) == 0
            ]
        else:
            self.valid_sample_idx_list = [i for i in range(len(self._mv_list))]

        self.len = len(self.valid_sample_idx_list)
        logger.warning(
            f"{self.name} {self.split_mode}_{self.data_split} Init Done. "
            f"Skip frames: {self.skip_frames}. Total {self.len} samples"
        )

    def __len__(self):
        return self.len

    def __getitem__(self, sample_idx):
        sample_idx = self.valid_sample_idx_list[sample_idx]
        idx_list = self._mv_list[sample_idx]

        # load all samples
        sample = {}
        sample["sample_idx"] = idx_list
        sample["cam_extr"] = []
        sample["target_cam_extr"] = []
        # sample["_image"] = []
        sample["_hand_side_ori"] = []

        # reorder idx_list to use allocentric_top as 0
        idx_list = [idx_list[3], idx_list[0], idx_list[1], idx_list[2]]
        for internal_idx in idx_list:
            internal_label = self._dataset[internal_idx]
            hand_side = internal_label["hand_side"]
            # print(hand_side)
            if hand_side == "right":
                sample["cam_extr"].append(internal_label["cam_extr"])
                sample["target_cam_extr"].append(internal_label["cam_extr"])
            else:
                cam_extr_flip = flip_cam_extr(internal_label["cam_extr"])
                sample["cam_extr"].append(cam_extr_flip)
                sample["target_cam_extr"].append(cam_extr_flip)
            sample["_hand_side_ori"].append(hand_side)
            # sample["_image"].append(internal_label["_image"])

            for query in [
                "rot_rad",
                "rot_mat3d",
                "affine",
                "image",
                "target_bbox_center",
                "target_bbox_scale",
                "target_joints_2d",
                "target_joints_vis",
                # "target_root_d",
                # "target_joints_uvd",
                # "target_verts_uvd",
                "affine_postrot",
                "target_cam_intr",
                "target_joints_3d",
                "target_verts_3d",
                # "target_joints_3d_rel",
                # "target_verts_3d_rel",
                # "target_root_joint",
                "target_joints_3d_no_rot",
                "target_verts_3d_no_rot",
                "idx",
                "cam_center",
                "bbox_center",
                "bbox_scale",
                "cam_intr",
                "joints_2d",
                "joints_3d",
                "verts_3d",
                "joints_vis",
                "joints_uvd",
                "verts_uvd",
                "image_path",
                "raw_size",
                "image_mask",
                "extr_prerot",
                # 'target_joints_heatmap',
                "cam_serial",
            ]:
                _value = internal_label[query]
                if query not in sample:
                    sample[query] = [_value]
                else:
                    sample[query].append(_value)

        if self.master_system == "as_constant_camera":
            # Here the as_constant_camera asserts master_id = 0
            master_id = 0  # MAGIC NUMBER: use allocentric_top
            T_master_2_new_master = sample["target_cam_extr"][master_id].copy()
            master_joints_3d = sample["target_joints_3d_no_rot"][master_id]
            master_verts_3d = sample["target_verts_3d_no_rot"][master_id]
            master_serial = sample["cam_serial"][master_id]
        else:
            pass  # should not go here

        for i, T_m2c in enumerate(sample["target_cam_extr"]):
            # we request inverse to be extr here (tf from camspace to master instead of tf from master to camspace)
            # i.e. (Extr @ p == p_master)
            # T_new_master_2_cam = np.linalg.inv(T_m2c @ np.linalg.inv(T_master_2_new_master))
            extr_prerot = sample["extr_prerot"][i]
            extr_prerot_tf_inv = np.eye(4).astype(extr_prerot.dtype)
            extr_prerot_tf_inv[:3, :3] = extr_prerot.T
            T_new_master_2_cam = T_master_2_new_master @ np.linalg.inv(T_m2c)
            # rotate the point then transform to new master
            # note we request the inverse to be extr here (tf from camspace 2 master)
            # (Extr @ R_aug_inv @ p == p_master)
            sample["target_cam_extr"][i] = T_new_master_2_cam @ extr_prerot_tf_inv

        for query in sample.keys():
            if isinstance(sample[query][0], (int, float, np.ndarray, torch.Tensor)):
                sample[query] = np.stack(sample[query])

        sample["master_id"] = master_id
        sample["master_joints_3d"] = master_joints_3d
        sample["master_verts_3d"] = master_verts_3d
        sample["master_serial"] = master_serial

        # Added raw from dexycb.py/ho3d.py
        if self.random_n_views:
            assert sample["master_id"] == 0  # ! The master must be the first.
            masked_sample = {}
            indices = [i for i in range(1, self.n_views)]  # idx from 1 to n - 1
            num_views_keep = random.randint(self.min_views - 1, self.n_views - 1)
            sample_idx_keep = random.sample(indices, num_views_keep)
            sample_idx_keep.append(0)  # always keep the master

            for key, value in sample.items():
                if (
                    not isinstance(value, int) and len(value) == self.n_views
                ):  # process info with len == N_VIEW
                    masked_value = np.array([value[i] for i in sample_idx_keep])
                    masked_sample[key] = masked_value
                else:
                    masked_sample[key] = value  # other info with len != N_VIEW
            sample = masked_sample

        # TODO: Add the random cam_view num here (referring to dexycb.py)
        # Currently, there is no key <cam_serial> and as_first_camera

        return sample


def lookat_to_extr(eye, target, up):
    # Normalize vectors
    def normalize(v):
        return v / np.maximum(np.linalg.norm(v), np.finfo(v.dtype).eps)

    # Calculate forward, right, and up vectors
    f = normalize(target - eye)
    r = normalize(np.cross(f, up))
    u = normalize(np.cross(r, f))

    # Create the rotation matrix
    R = np.column_stack([r, u, -f])

    # Construct the extrinsic matrix - note the use of .T for correct orientation
    extr = np.identity(4)
    extr[:3, :3] = R.T
    extr[:3, 3] = np.dot(R.T, -eye)

    return extr


def extr_to_lookat(extrinsic_matrix):
    # Extract rotation matrix and translation vector
    R = extrinsic_matrix[:3, :3].T  # Transpose back to get original R
    T = extrinsic_matrix[:3, 3]

    # Calculate eye position
    eye = -np.dot(R, T)

    # Reconstruct forward, right, and up vectors
    r = R[:, 0]
    u = R[:, 1]
    f = -R[:, 2]

    # Target is a point along the forward direction from the eye
    target = eye + f
    # Up vector is simply the up vector used to construct R
    up = u

    return eye.copy(), target.copy(), up.copy()


def flip_cam_extr(cam_extr):
    eye, target, up = extr_to_lookat(cam_extr)
    # print(eye, target, up)
    # up_point = eye + up
    eye[0] = -eye[0]
    target[0] = -target[0]
    up[0] = -up[0]
    res = lookat_to_extr(eye, target, up)
    return res
