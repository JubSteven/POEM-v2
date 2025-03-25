import hashlib
import json
import os
import pickle
import random
import time
import warnings
from collections import defaultdict
from typing import List

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import yaml
from manotorch.manolayer import ManoLayer, MANOOutput
from termcolor import colored
from ..utils.collation import key_filter
from ..utils.builder import DATASET
from ..utils.config import CN
from ..utils.etqdm import etqdm
from ..utils.logger import logger
from ..utils.transform import (SE3_transform, aa_to_rotmat, batch_ref_bone_len, cal_transform_mean, denormalize,
                               get_annot_center, get_annot_scale, persp_project, rotmat_to_aa)
from .hdata import HDataset, kpId2vertices
from .ho3d import HO3Dv3MultiView, HO3D


class HO3D_Test(HO3D):

    def __init__(self, cfg):
        super().__init__(cfg)

        assert self.data_split == "test"

        with open(os.path.join(self.data_root, "HO3D_v3_official_gt/evaluation_xyz.json"), "r") as f:
            evaluation_xyz = json.load(f)
        with open(os.path.join(self.data_root, "HO3D_v3_official_gt/evaluation_verts.json"), "r") as f:
            evaluation_verts = json.load(f)

        self.evaluation_xyz = np.array(evaluation_xyz, dtype=np.float32)
        self.evaluation_verts = np.array(evaluation_verts, dtype=np.float32)

    def get_joints_3d(self, idx):
        # seq, img_idx = self.seq_idx[idx]
        # annot = self.annot_mapping[seq][img_idx]
        joints_3d = self.evaluation_xyz[idx]
        joints_3d = self.cam_extr[:3, :3].dot(joints_3d.transpose()).transpose()
        joints_3d = joints_3d[self.reorder_idxs]
        return joints_3d.astype(np.float32)

    def get_verts_3d(self, idx):
        handverts = self.evaluation_verts[idx]
        transf_handverts = self.cam_extr[:3, :3].dot(handverts.transpose()).transpose()
        return transf_handverts.astype(np.float32)


@DATASET.register_module()
class HO3Dv3MultiViewTest(HO3Dv3MultiView):

    def __init__(self, cfg):
        self.name = type(self).__name__
        self.cfg = cfg
        self.n_views = cfg.N_VIEWS
        self.data_split = cfg.DATA_SPLIT
        assert self.data_split == "test", f"{self.name} only support test data split"

        self.random_n_views = cfg.RANDOM_N_VIEWS  # whether to truncate the batch into size min_views to N_VIEWS
        self.min_views = 0 if not self.random_n_views else cfg.MIN_VIEWS  # minimum views required
        self.return_before_aug = cfg.get("RETURN_BEFORE_AUG", False)
        self.n_views_kept = cfg.get("N_VIEWS_KEPT", self.n_views)
        self.filter_keys = cfg.get("FILTER_KEYS", False)
        assert self.n_views_kept <= self.n_views, f"n_views_kept must be less than or equal to n_views"
        assert self.data_split in ["train", "val", "test"], f"{self.name} unsupport data split {self.data_split}"

        self.master_system = cfg.MASTER_SYSTEM
        assert self.master_system in ["as_constant_camera", "as_first_camera"], \
            f"{self.name} only support as_constant_camera master system"
        self.const_cam_id = cfg.CONST_CAM_ID

        self.data_mode = cfg.DATA_MODE
        assert self.data_mode == "3D", f"{self.name} only support 3D data mode"
        self.split_mode = cfg.SPLIT_MODE
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        _, _testset = self._single_view_ho3d()

        self.set_mappings = {f"{cfg.SPLIT_MODE}_train": None, f"{cfg.SPLIT_MODE}_test": _testset}
        self.root = _testset.root
        if self.data_split == "train":
            assert False
        elif self.data_split == "test":
            pass

        # 10: side_view_facing_whiteboard
        # 11: top_view_facing_desk
        # 12: front_view_facing_wall
        # 13: side_view_facing_screens
        # 14: ego_view_facing_desk
        self.view_name = [
            'side_view_facing_whiteboard',
            'top_view_facing_desk',
            'front_view_facing_wall',
            'side_view_facing_screens',
            'ego_view_facing_desk',
        ]

        self.multivew_mapping = {}
        self.multiview_sample_idxs = []
        self.multiview_sample_infos = []

        if self.split_mode in ['paper', 'v2', 'v3']:  # full view mode
            info_path_eval = os.path.join(self.root, "evaluation.txt")
            with open(info_path_eval, "r") as f:
                lines = f.readlines()
            seq_frames = [line.strip().split("/") for line in lines]
            self._mapping_multiview(seq_frames=seq_frames)
            self._mapping_idxs_infos()

        else:
            raise ValueError(f"{self.split_mode} is not supported")

        logger.warning(
            f"{self.name} {self.split_mode}_{self.data_split} Init Done. {len(self.multiview_sample_idxs)} samples")

    def _single_view_ho3d(self):
        cfg_test = dict(
            TYPE="HO3D",
            DATA_SPLIT="test",
            DATA_MODE=self.data_mode,
            SPLIT_MODE=self.split_mode,
            DATA_ROOT=self.cfg.DATA_ROOT,
            TRANSFORM=self.cfg.TRANSFORM,
            DATA_PRESET=self.cfg.DATA_PRESET,
        )

        cfg_test["DATA_SPLIT"] = "test"
        cfg_test["USE_GT_FROM_MULTIVIEW"] = False

        ho3d_test = HO3D_Test(CN(cfg_test))

        return None, ho3d_test

    def _mapping_multiview(self, seq_frames):
        for i, (seq, frame_idx) in enumerate(etqdm(seq_frames)):

            if seq == "SM1":
                seq_name_main = "SM1"
                cam_id = None
            else:
                seq_name_main = seq[:-1]
                cam_id = int(seq[-1])

            if (seq_name_main, frame_idx) not in self.multivew_mapping:
                self.multivew_mapping[(seq_name_main, frame_idx)] = [(cam_id, i)]
            else:
                self.multivew_mapping[(seq_name_main, frame_idx)].append((cam_id, i))

    def _mapping_idxs_infos(self):
        # 10: side_view_facing_whiteboard
        # 11: top_view_facing_desk
        # 12: front_view_facing_wall
        # 13: side_view_facing_screens
        # 14: ego_view_facing_desk
        for key, value in self.multivew_mapping.items():
            seq_name_main, frame_idx = key

            self.multiview_sample_idxs.append([i for (_, i) in value])
            self.multiview_sample_infos.append([{
                "set_name": "paper_test",
                "seq_name": seq_name_main + f"{cam_id}" if cam_id is not None else seq_name_main,
                "seq_name_main": seq_name_main,
                "cam_id": cam_id,
                "frame_idx": frame_idx,
                "subfolder": "evaluation",
                "view_name": self.view_name[cam_id] if cam_id is not None else None,
            } for (cam_id, _) in value])

    def __len__(self):
        return len(self.multiview_sample_idxs)

    def __getitem__(self, idx):
        multiview_id_list = self.multiview_sample_idxs[idx]
        multiview_info_list = self.multiview_sample_infos[idx]

        seq_name_main = multiview_info_list[0]["seq_name_main"]

        if multiview_info_list[0]["cam_id"] is not None:
            extr_mapping = {}
            for cam_id in range(self.n_views):
                # cam_id = int(multiview_info_list[i]["cam_id"])
                extr_seq_dir = os.path.join(self.root, "calibration", seq_name_main, "calibration",
                                            f"trans_{cam_id}.txt")
                with open(extr_seq_dir) as f:
                    extr = np.loadtxt(f, dtype=np.float32)
                extr_mapping[cam_id] = extr

            # get true cam_id, the cam_id above just for convenient index
            true_cam_order_dir = os.path.join(self.root, "calibration", seq_name_main, "calibration", "cam_orders.txt")
            true_cam_orders = [int(float(number)) for line in open(true_cam_order_dir, 'r') for number in line.split()]

        sample = dict()
        sample["sample_idx"] = multiview_id_list
        sample["cam_extr"] = list()
        sample["cam_serial"] = list()
        for i, info in zip(multiview_id_list, multiview_info_list):
            # get the source set -- one of the  HO3D s#_train, s#_val and s#_test.
            source_set = self.set_mappings[info["set_name"]]
            if info["seq_name"] == "SM1":
                T_master_2_cam = np.eye(4)
            else:
                cam_id = int(info["seq_name"][-1])
                # change 2 the true cam_id
                if cam_id == 0:
                    cam_id = true_cam_orders[0]
                elif cam_id == 1:
                    cam_id = true_cam_orders[1]
                elif cam_id == 2:
                    cam_id = true_cam_orders[2]
                elif cam_id == 3:
                    cam_id = true_cam_orders[3]
                elif cam_id == 4:
                    cam_id = true_cam_orders[4]
                T_master_2_cam = extr_mapping[cam_id]

            sample["cam_extr"].append(T_master_2_cam)

            cam_serial = info["seq_name"]
            sample["cam_serial"].append(cam_serial)

            # get sample from the source set. (HO3D's getitem)
            src_sample = source_set[i]  # @NOTE: __getitem__ here !
            for query, value in src_sample.items():
                if query in sample:
                    sample[query].append(value)
                else:
                    sample[query] = [value]

        # >>>>>>>>>>>
        # you can use self._testing() to test here
        # <<<<<<<<<<<

        # @FLAG dump
        if self.return_before_aug:
            return sample

        # set a new master
        # if self.master_system == "as_first_camera":
        #     new_master_id = 0
        #     new_master_serial = sample["cam_serial"][new_master_id]
        #     T_master_2_new_master = sample["cam_extr"][new_master_id]
        #     master_joints_3d = sample["target_joints_3d_no_rot"][new_master_id]
        #     master_verts_3d = sample["target_verts_3d_no_rot"][new_master_id]

        # elif self.master_system == "as_constant_camera":
        #     new_master_serial = sample["cam_serial"][0][:-1] + f"{self.const_cam_id}"
        #     new_master_id = sample["cam_serial"].index(new_master_serial)
        #     T_master_2_new_master = sample["cam_extr"][new_master_id]
        #     master_joints_3d = sample["target_joints_3d_no_rot"][new_master_id]
        #     master_verts_3d = sample["target_verts_3d_no_rot"][new_master_id]

        new_master_serial = random.sample(sample["cam_serial"], 1)[0]
        new_master_id = sample["cam_serial"].index(new_master_serial)
        T_master_2_new_master = sample["cam_extr"][new_master_id]
        master_joints_3d = sample["target_joints_3d_no_rot"][new_master_id]
        master_verts_3d = sample["target_verts_3d_no_rot"][new_master_id]

        sample["target_cam_extr"] = list()
        for i, T_m2c in enumerate(sample["cam_extr"]):
            T_new_master_2_cam = np.linalg.inv(T_master_2_new_master) @ T_m2c
            extr_prerot = sample["extr_prerot"][i]  # (3, 3)
            extr_pre_transf = np.concatenate([extr_prerot, np.zeros((3, 1))], axis=1)
            extr_pre_transf = np.concatenate([extr_pre_transf, np.array([[0, 0, 0, 1]])], axis=0)

            T_new_master_2_cam = np.linalg.inv(extr_pre_transf @ np.linalg.inv(T_new_master_2_cam))
            sample["target_cam_extr"].append(T_new_master_2_cam.astype(np.float32))

        sample.pop("image_full")
        for query in sample.keys():
            if isinstance(sample[query][0], (int, float, np.ndarray, torch.Tensor)):
                sample[query] = np.stack(sample[query])

        sample["master_id"] = new_master_id
        sample["master_serial"] = new_master_serial
        sample["master_joints_3d"] = master_joints_3d
        sample["master_verts_3d"] = master_verts_3d
        sample["cam_serial"] = list(sample["cam_serial"])  # normalization for further collation

        if self.random_n_views:
            assert sample["master_id"] == 0  # ! The master must be the first.
            masked_sample = {}
            indices = [i for i in range(1, self.n_views)]  # idx from 1 to n - 1
            if self.n_views_kept != self.n_views:
                num_views_keep = self.n_views_kept - 1
            else:
                num_views_keep = random.randint(self.min_views - 1, self.n_views - 1)
            sample_idx_keep = random.sample(indices, num_views_keep)
            sample_idx_keep.insert(0, 0)  # always keep the master

            for key, value in sample.items():
                if not isinstance(value, int) and len(value) == self.n_views:  # process info with len == 8
                    masked_value = np.array([value[i] for i in sample_idx_keep])
                    masked_sample[key] = masked_value
                else:
                    masked_sample[key] = value  # other info with len != 8
            sample = masked_sample

        if self.filter_keys:
            sample = key_filter(sample)

        return sample
