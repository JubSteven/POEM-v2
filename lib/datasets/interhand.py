import hashlib
import json
import os
import pickle
import random
import warnings
from typing import List

import time
import imageio
import numpy as np
import torch
from manotorch.manolayer import ManoLayer, MANOOutput
from termcolor import colored
from tqdm import tqdm

from ..utils.collation import key_filter
from ..utils.builder import DATASET
from ..utils.config import CN
from ..utils.logger import logger
from ..utils.transform import (batch_ref_bone_len, aa_to_rotmat, cal_transform_mean, get_annot_center,
                               get_annot_scale, persp_project, rotmat_to_aa)
from .hdata import HDataset, kpId2vertices


@DATASET.register_module()
class InterHand(HDataset):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.name = "InterHand"
        self.data_split = cfg.DATA_SPLIT # train, val, test
        self.joint_num = 21 # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
        self.data_mode = cfg.DATA_MODE
        assert self.data_mode == "3D", f"{self.name} only support 3D data mode"
        self.center_idx = self.cfg.DATA_PRESET.CENTER_IDX
        assert self.center_idx == 0
        
        self.skip_frames = cfg.get("SKIP_FRAMES", 0)        
        assert self.data_split in ["train", "val", "test"], f"{self.name} unsupport data split {self.data_split}"
        self.use_left_hand = cfg.USE_LEFT_HAND
        assert self.use_left_hand == False
        
        self._interhand_mano_right = ManoLayer(
            flat_hand_mean=False,
            side="right",
            mano_assets_root="assets/mano_v1_2",
            center_idx=self.center_idx
        )
        
        self._interhand_mano_left = ManoLayer(
            flat_hand_mean=False,
            side="left",
            mano_assets_root="assets/mano_v1_2",
            center_idx=self.center_idx
        )
        
        self.load_dataset()
        
    def _preload(self):
        self.root = os.path.join(self.data_root, self.name)
        os.environ["INTER_HAND_DIR"] = self.root
        
        self.img_path = os.path.join(self.data_root, "InterHand", "images")
        self.annot_path = os.path.join(self.data_root, "InterHand", "anno_packed")

        self.sample_idxs_path = os.path.join(self.annot_path, self.data_split, "index.pkl")
        with open(self.sample_idxs_path, "rb") as p_f:
            self.sample_idxs = pickle.load(p_f)


    def load_dataset(self):
        self._preload()
        
        interhand_name = f"{self.data_split}"
        logger.info(f"Interhand use split: {interhand_name}")        
        logger.info(f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])}"
                    f" samples for data_split {self.data_split}")
        

    def __len__(self):
        return len(self.sample_idxs)
    
    def get_sample_idx(self):
        return self.sample_idxs
    
    def load_sample(self, idx):
        aid = self.sample_idxs[idx] # Get original idx -> new idx
        annot_idx_path = os.path.join(self.annot_path, self.data_split, "{}.pkl".format(aid))
        with open(annot_idx_path, "rb") as p_f:
            sample = pickle.load(p_f)
        return sample
    
    def get_image(self, idx):
        path = self.get_image_path(idx)
        img = np.array(imageio.imread(path, pilmode="RGB"), dtype=np.uint8)
        return img
    
    def get_image_path(self, idx):
        sample = self.load_sample(idx)
        return sample["img_path"]
    
    def get_image_mask(self, idx):
        # Not implemented?
        raise NotImplementedError(f"{self.name} does not support get_image_mask")

    def get_joints_3d(self, idx):
        sample = self.load_sample(idx)
        raw_joints_3d = sample["joint_cam_coord"][:21] / 1000 # In meter
        joints_3d = raw_joints_3d[[20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16], :] # Rearrage it in the conventional order
        return joints_3d
    
    
    def get_verts_3d(self, idx):
        pose = torch.from_numpy(self.get_mano_pose(idx)).unsqueeze(0)
        shape = torch.from_numpy(self.get_mano_shape(idx)).unsqueeze(0)
        
        mano_layer = self._interhand_mano_left if self.use_left_hand else self._interhand_mano_right
        mano_out: MANOOutput = mano_layer(pose, shape)
        
        hand_verts = mano_out.verts[0].numpy() + self.get_joints_3d(idx)[0] # ignore trans anno and use 3d joints instead (wrist coord is 0 after modification)
        
        return hand_verts.astype(np.float32)
        

    def get_verts_uvd(self, idx):
        v3d = self.get_verts_3d(idx)
        intr = self.get_cam_intr(idx)
        uv = persp_project(v3d, intr)[:, :2]
        d = v3d[:, 2:]  # (778, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_bone_scale(self, idx):
        joints_3d = self.get_joints_3d(idx)
        bone_len = batch_ref_bone_len(np.expand_dims(joints_3d, axis=0)).squeeze(0)
        return bone_len.astype(np.float32)

    def get_joints_2d(self, idx):
        # No keypoints in 2D
        joints_3d = self.get_joints_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return persp_project(joints_3d, cam_intr)

    def get_joints_uvd(self, idx):
        uv = self.get_joints_2d(idx)
        d = self.get_joints_3d(idx)[:, 2:]  # (21, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_cam_intr(self, idx):
        sample = self.load_sample(idx)
        focal = sample["focal"]
        princpt = sample["princpt"]
        return np.array(
            [
                [focal[0], 0.0, princpt[0]],
                [0.0, focal[1], princpt[1]],
                [0.0, 0.0, 1.0]
            ]
        )
        
    def get_cam_extr(self, idx):
        sample = self.load_sample(idx)
        rot = sample["camrot"]
        trans = sample["campos"] / 1000
        trans = -np.dot(rot, trans.reshape(3, 1)).reshape(3) # -Rt -> t
        return np.concatenate([rot, trans.reshape(3, 1)], axis=1)

    def get_cam_center(self, idx):
        intr = self.get_cam_intr(idx)
        return np.array([intr[0, 2], intr[1, 2]])

    def get_sides(self, idx):
        return "right"

    def get_bbox_center_scale(self, idx):
        joints2d = self.get_joints_2d(idx)  # (21, 2)
        center = get_annot_center(joints2d)
        scale = get_annot_scale(joints2d)
        return center, scale

    def get_sample_identifier(self, idx):
        sample = self.load_sample(idx)
        original_idx = sample["idx"]
        res = f"{self.name}_{original_idx}"
        return res
    
    def get_mano_pose(self, idx):
        sample = self.load_sample(idx)
        pose_m = np.array(sample["pose"], dtype=np.float32)
        
        root, remains = pose_m[:3], pose_m[3:]
        root = rotmat_to_aa(sample["camrot"] @ aa_to_rotmat(root))
        handpose_transformed = np.concatenate((root, remains), axis=0)
        return handpose_transformed

    def get_mano_shape(self, idx):
        sample = self.load_sample(idx)
        shape = sample["shape"]
        shape = np.array(shape, dtype=np.float32)
        return shape

    def get_rawimage_size(self, idx):
        img = self.get_image(idx)
        return [img.shape[0], img.shape[1]]
        # return self.inter_hand[idx]["image"].shape


@DATASET.register_module()
class InterHandMultiView(torch.utils.data.Dataset):
    
    def __init__(self, cfg):
        self.name = type(self).__name__
        self.cfg = cfg
        self.n_views = cfg.N_VIEWS
        self.data_split = cfg.DATA_SPLIT
        self.random_n_views = cfg.RANDOM_N_VIEWS # whether to truncate the batch into size min_views to N_VIEWS
        self.min_views = 0 if not self.random_n_views else cfg.MIN_VIEWS # minimum views required
        self.skip_frames = cfg.get("SKIP_FRAMES", 0)
        self.filter_keys = cfg.get("FILTER_KEYS", False)
        self.n_views_kept = cfg.get("N_VIEWS_KEPT", self.n_views)
        assert self.n_views_kept <= self.n_views, f"n_views_kept must be less than or equal to n_views"
        
        self.master_system = cfg.MASTER_SYSTEM
        assert self.master_system in [
            "as_first_camera"
        ], f"{self.name} only support as_first_camera master system"
        
        self.data_mode = cfg.DATA_MODE
        assert self.data_mode == "3D", f"{self.name} only support 3D data mode"
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX # As specified in the single hand dataset
        
        self.return_before_aug = cfg.get("RETURN_BEFORE_AUG", False)

        self.CONST_CAM_ID = 410028 # ! Could be modified. This is set randomly from a camera that is used in all multiview data
        
        _trainset, _valset, _testset = self._single_view_interhand()
        
        self.set_mappings = {
            "train": _trainset,
            "val": _valset,
            "test": _testset
        }
        self.root = _trainset.root
        

        self._preload()
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as p_f:
                self.annotation = pickle.load(p_f)
                self.multiview_sample_idxs = self.annotation['idx']
                self.multiview_sample_infos = self.annotation['info']
            logger.info(f"Loaded cache for {self.name}_{self.data_split} from {self.cache_path}")
        else:
            self.multiview_sample_idxs = []
            self.multiview_sample_infos = []
            
            # source set is of type InterHand
            source_set: InterHand = self.set_mappings[self.data_split]
            multiview_mapping = {}
            bar = tqdm(range(len(source_set.sample_idxs)))
            for idx, _ in enumerate(bar):
                # capture_id and frame_id -> a scene
                # put all camera views within a scene together as follows
                sample = source_set.load_sample(idx)
                capture_id = sample["capture"]
                frame_id = sample["frame"]
                cam_id = sample["cam"]
                if (capture_id, frame_id) not in multiview_mapping:
                    multiview_mapping[(capture_id, frame_id)] = [(cam_id, idx)]
                else:
                    multiview_mapping[(capture_id, frame_id)].append((cam_id, idx))

            for key, value in multiview_mapping.items():
                capture_id, frame_id = key
                self.multiview_sample_idxs.append([i for (_, i) in value])
                self.multiview_sample_infos.append([{
                    "set_name": self.data_split,
                    "seq_id": capture_id,
                    "cam_id": cam_id,
                    "frame_id": frame_id,
                } for (cam_id, _) in value])
                
            self.annotation = {
                "idx": self.multiview_sample_idxs,
                "info": self.multiview_sample_infos
            }
            
                
            if os.path.exists(os.path.dirname(self.cache_path)) is False:
                os.makedirs(os.path.dirname(self.cache_path))
                
            with open(self.cache_path, "wb") as p_f:
                pickle.dump(self.annotation, p_f)
            logger.info(f"Saved cache for {self.name}_{self.data_split} to {self.cache_path}")
            
        total_len = len(self.multiview_sample_idxs)
        if self.skip_frames != 0:
            self.valid_sample_idx_list = [i for i in range(total_len) if i % (self.skip_frames + 1) == 0]
        else:
            self.valid_sample_idx_list = [i for i in range(total_len)]

        self.length = len(self.valid_sample_idx_list)
        logger.warning(f"{self.name} {self.data_split} Init Done. "
                       f"Skip frames: {self.skip_frames}, total {self.length} samples")
    
    def _single_view_interhand(self):
        
        cfg_train = dict(
            TYPE="InterHand",
            DATA_SPLIT="train",
            DATA_MODE=self.data_mode,
            DATA_ROOT=self.cfg.DATA_ROOT,
            USE_LEFT_HAND=self.cfg.USE_LEFT_HAND,
            FILTER_INVISIBLE_HAND=self.cfg.FILTER_INVISIBLE_HAND,
            TRANSFORM=self.cfg.TRANSFORM,
            DATA_PRESET=self.cfg.DATA_PRESET,
        )

        cfg_val, cfg_test = cfg_train.copy(), cfg_train.copy()
        cfg_val["DATA_SPLIT"] = "val"
        cfg_test["DATA_SPLIT"] = "test"

        interhand_train = InterHand(CN(cfg_train))
        interhand_val = InterHand(CN(cfg_val))
        interhand_test = InterHand(CN(cfg_test))

        return interhand_train, interhand_val, interhand_test
    
    def _preload(self):
        self.name = "InterHandMultiView"

        self.cache_identifier_dict = {
            "data_split": self.data_split,
            "use_left_hand": self.cfg.USE_LEFT_HAND,
            "filter_invisible_hand": self.cfg.FILTER_INVISIBLE_HAND,
        }
        
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format(self.cache_identifier))
    
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        idx = self.valid_sample_idx_list[idx]
        multiview_id_list = self.multiview_sample_idxs[idx]
        multiview_info_list = self.multiview_sample_infos[idx]
        total_cam_num = len(multiview_info_list)

        if self.master_system == "as_first_camera":
            # NOTE: shuffle the order of cameras to make the ``first camera`` in training mode not always the same
            if self.data_split == "train":
                lists_to_shuffle = list(zip(multiview_id_list, multiview_info_list))
                random.shuffle(lists_to_shuffle)
                multiview_id_list, multiview_info_list = zip(*lists_to_shuffle)
                        
        else:
            # Should never be here
            raise NotImplementedError(f"Unknown master system {self.master_system}")
            
        sample = dict()
        sample["sample_idx"] = multiview_id_list
        sample["cam_extr"] = list()
        sample["cam_id"] = list()
        
        for i, info in zip(multiview_id_list, multiview_info_list):
            # get the source set -- one of the  DexYCB s#_train, s#_val and s#_test.
            source_set = self.set_mappings[info["set_name"]]
            cam_id = info["cam_id"]
            sample["cam_id"].append(cam_id)

            # get sample from the single view source set. (Interhand's __getitem__)
            src_sample = source_set[i]
            for query, value in src_sample.items():
                if query in sample:
                    sample[query].append(value)
                else:
                    sample[query] = [value]
        
            raw_T = source_set.get_cam_extr(i)
            T_master_2_cam = np.asarray(raw_T, dtype=np.float32).reshape(3, 4)
            T_master_2_cam = np.concatenate([T_master_2_cam, np.array([[0, 0, 0, 1]])], axis=0).astype(np.float32)
            sample["cam_extr"].append(T_master_2_cam)
            
        # @FLAG dump
        if self.return_before_aug:
            sample["cam_serial"] = ["cam{}".format(i) for i in sample["cam_id"]]
            return sample
            
        new_master_id = 0
        T_master_2_new_master = sample["cam_extr"][new_master_id]
        master_joints_3d = sample["target_joints_3d_no_rot"][new_master_id]
        master_verts_3d = sample["target_verts_3d_no_rot"][new_master_id]

        sample.pop("image_full")

        sample["target_cam_extr"] = list()
        for i, T_m2c in enumerate(sample["cam_extr"]):
            extr_prerot = sample["extr_prerot"][i]
            extr_prerot_tf_inv = np.eye(4).astype(extr_prerot.dtype)
            extr_prerot_tf_inv[:3, :3] = extr_prerot.T
            T_new_master_2_cam = T_master_2_new_master @ np.linalg.inv(T_m2c)
            sample["target_cam_extr"].append(T_new_master_2_cam @ extr_prerot_tf_inv)

        for query in sample.keys():
            if isinstance(sample[query][0], (int, float, np.ndarray, torch.Tensor)):
                sample[query] = np.stack(sample[query])

        sample["master_id"] = new_master_id
        sample["master_joints_3d"] = master_joints_3d
        sample["master_verts_3d"] = master_verts_3d
        
        if self.random_n_views:
            assert sample["master_id"] == 0 # ! The master must be the first.
            masked_sample = {}
            indices = [i for i in range(1, total_cam_num)] # idx from 1 to n - 1
            if self.n_views_kept != self.n_views:
                num_views_keep = self.n_views_kept - 1
            else:
                num_views_keep = random.randint(self.min_views-1, self.n_views-1)
            sample_idx_keep = random.sample(indices, num_views_keep)
            sample_idx_keep.insert(0, 0) # always keep the master

            for key, value in sample.items():
                if not isinstance(value, int) and len(value) == total_cam_num: # process info with len == 8
                    masked_value = np.array([value[i] for i in sample_idx_keep])
                    masked_sample[key] = masked_value
                else:
                    masked_sample[key] = value # other info with len != 8
            sample = masked_sample
            
        if self.filter_keys:
            sample = key_filter(sample)
        
        return sample