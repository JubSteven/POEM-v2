import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import List

import imageio
import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from termcolor import colored

from ..utils.builder import DATASET
from ..utils.etqdm import etqdm
from ..utils.logger import logger
from ..utils.transform import get_annot_center, get_annot_scale
from .hdata import HDataset, kpId2vertices


@DATASET.register_module()
class YT3D(HDataset):

    def __init__(self, cfg):
        super(YT3D, self).__init__(cfg)
        assert self.data_mode != "3D", "YT3D does not dupport 3D mode"
        assert self.data_split in ["train", "val", "test"], "YT3D data_split must be one of ['train', 'val', 'test']"
        self.rMANO = ManoLayer(side="right", mano_assets_root="assets/mano_v1_2")
        self.lMANO = ManoLayer(side="left", mano_assets_root="assets/mano_v1_2")
        self.J_regressor = self.rMANO.th_J_regressor
        self.rhand_faces = self.rMANO.get_mano_closed_faces().numpy()
        self.lhand_faces = self.lMANO.get_mano_closed_faces().numpy()

        self.load_dataset()
        logger.info(f"initialized child class: {self.name}")

    def _preload(self):
        self.name = "YouTube-3D-Hands"
        self.root = os.path.join(self.data_root, self.name)
        self.root_supp = os.path.join(self.data_root, f"{self.name}_supp")

        self.cache_identifier_dict = {
            "data_split": self.data_split,
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format(self.cache_identifier))

    def load_dataset(self):
        self._preload()
        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        self.img_path_id_mapping = {}
        if os.path.exists(self.cache_path) and self.use_cache:  # load from cache
            with open(self.cache_path, "rb") as p_f:
                annotations = pickle.load(p_f)
            logger.info(f"Loaded cache for {self.name}_{self.data_split} from {self.cache_path}")
        else:  # load from disk
            self.image_ids = []
            self.image_sizes = []
            self.image_paths = []
            self.mask_paths = []
            self.hand_sides = []
            self.verts_uvd_paths = []
            self.joints_uvd = []

            logger.info("Loading YT3D dataset from disk, this may take a while...")
            fp_data = os.path.join(self.root, f"youtube_{self.data_split}.json")
            with open(fp_data, "r") as file:
                raw_data = json.load(file)

            logger.info(f"YT3D {self.data_split} has {len(raw_data['annotations'])} samples")

            img_infos = raw_data["images"]
            img_idxs = [im['id'] for im in img_infos]
            bar = etqdm(range(len(raw_data['annotations'])))
            for i, _ in enumerate(bar):
                bar.set_description(f"load from disk {self.name} {self.data_split}")
                # process each data sample
                ann = raw_data['annotations'][i]
                img_id = ann["image_id"]
                img_info = img_infos[img_idxs.index(img_id)]
                img_path = img_info["name"].replace("youtube", "youtube_annotated")

                W = img_info["width"]
                H = img_info["height"]
                side = 'l' if ann["is_left"] == 1 else 'r'
                verts = np.array(ann["vertices"], dtype=np.float32)

                th_verts = torch.from_numpy(verts).float()
                joints = torch.matmul(self.J_regressor, th_verts)  # (16, 3)
                tipsId = [v[0] for k, v in kpId2vertices.items()]
                tips = th_verts[tipsId]  # (5, 3)
                joints = torch.cat([joints, tips], dim=0)
                # Reorder joints to match OpenPose definition
                joints = joints[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
                joints = joints.numpy()  # (21, 3)

                # path/to/frames/imgname.png --> imgname
                imgname = Path(img_path).stem
                # imgname --> imgname_datasplit_i
                dump_name = f"{imgname}_{self.data_split}_{side}_{i}"

                # .../frames/... --> .../verts_uvd/...
                verts_dump_path = img_path.replace("frames", "verts_uvd")
                # verts_uvd/imgname.png --> verts_uvd/imgname_datasplit_i.pkl
                verts_dump_path = verts_dump_path.replace(f"{imgname}.png", f"{dump_name}.pkl")
                verts_dump_path = os.path.join(self.root_supp, verts_dump_path)

                os.makedirs(os.path.dirname(verts_dump_path), exist_ok=True)
                if not os.path.exists(verts_dump_path):  # dump if not exists
                    with open(verts_dump_path, "wb") as p_f:
                        pickle.dump(verts, p_f)

                mask_path = img_path.replace("frames", "masks")
                mask_path = mask_path.replace(f"{imgname}.png", f"{dump_name}.png")

                img_load_path = os.path.join(self.root, img_path)
                mask_load_path = os.path.join(self.root_supp, mask_path)

                self.image_ids.append(img_id)
                self.image_sizes.append([W, H])
                self.image_paths.append(img_load_path)
                self.mask_paths.append(mask_load_path)
                self.verts_uvd_paths.append(verts_dump_path)
                self.joints_uvd.append(joints)
                self.hand_sides.append(side)

            annotations = {
                "image_ids": self.image_ids,
                "cache_identifier_dict": self.cache_identifier_dict,
                "image_paths": self.image_paths,
                "mask_paths": self.mask_paths,
                "verts_uvd_paths": self.verts_uvd_paths,
                "joints_uvd": self.joints_uvd,
                "hand_sides": self.hand_sides,
                "image_sizes": self.image_sizes,
            }

            with open(self.cache_path, "wb") as fid:
                pickle.dump(annotations, fid)
            logger.info(f"Wrote cache for {self.name}_{self.data_split} to {self.cache_path}")

        self.image_ids = annotations["image_ids"]
        self.image_paths = annotations["image_paths"]
        self.mask_paths = annotations["mask_paths"]
        self.verts_uvd_paths = annotations["verts_uvd_paths"]
        self.hand_sides = annotations["hand_sides"]
        self.joints_uvd = annotations["joints_uvd"]
        self.image_sizes = annotations["image_sizes"]

        self.n_samples = len(self.image_paths)
        self.sample_idxs = list(range(self.n_samples))

        logger.info(f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])}"
                    f"/{self.n_samples} samples for data_split {self.data_split}")

    def __len__(self):
        return len(self.sample_idxs)

    def get_sample_idx(self) -> List[int]:
        return self.sample_idxs

    def get_image(self, idx):
        path = self.image_paths[idx]
        image = np.array(imageio.imread(path, pilmode="RGB"), dtype=np.uint8)
        return image

    def get_image_path(self, idx):
        return self.image_paths[idx]

    def get_rawimage_size(self, idx):
        imsize = self.image_sizes[idx]
        # MUST (W, H)
        return imsize

    def get_image_mask(self, idx):
        path = self.mask_paths[idx]
        mask = np.array(imageio.imread(path, as_gray=True), dtype=np.uint8)
        return mask

    def get_cam_intr(self, idx):
        raise NotImplementedError(f"{self.name} does not support camera intrinsics")

    def get_cam_center(self, idx):
        imsize = self.image_sizes[idx]
        # (W/2, H/2)
        return np.array([imsize[0] // 2, imsize[1] // 2])

    def get_hand_faces(self, idx):
        hand_side = self.hand_sides[idx]
        return self.rhand_faces if hand_side == "r" else self.lhand_faces

    def get_joints_3d(self, idx):
        raise NotImplementedError(f"{self.name} does not support 3D joints")

    def get_verts_3d(self, idx):
        raise NotImplementedError(f"{self.name} does not support 3D verts")

    def get_joints_2d(self, idx):
        joints_2d = self.joints_uvd[idx][:, :2].copy()  # (21, 2)
        return joints_2d

    def get_joints_uvd(self, idx):
        juvd = self.joints_uvd[idx].copy()  # (21, 3)
        juvd[:, 2] = juvd[:, 2] / 1000.0  # to meter
        return juvd

    def get_verts_uvd(self, idx):
        verts_uvd_path = self.verts_uvd_paths[idx]
        with open(verts_uvd_path, "rb") as p_f:
            verts_uvd = pickle.load(p_f)  # (778, 3)

        verts_uvd[:, 2] = verts_uvd[:, 2] / 1000.0  # to meter
        return verts_uvd

    def get_verts_2d(self, idx):
        verts_2d = self.get_verts_uvd(idx)[:, :2]  # (778, 2)
        return verts_2d

    def get_sides(self, idx):
        hand_side = self.hand_sides[idx]
        return "right" if hand_side == 'r' else "left"

    def get_bone_scale(self, idx):
        raise NotImplementedError(f"{self.name} does not support bone scale")

    def get_bbox_center_scale(self, idx):
        joints_2d = self.get_joints_2d(idx)
        center = get_annot_center(joints_2d)
        scale = get_annot_scale(joints_2d)
        return center, scale

    def get_mano_pose(self, idx):
        raise NotImplementedError(f"{self.name} does not support mano pose")

    def get_mano_shape(self, idx):
        raise NotImplementedError(f"{self.name} does not support mano shape")

    def get_sample_identifier(self, idx):
        res = f"{self.name}__{self.cache_identifier_raw}__{idx}"
        return res

    def getitem_3d(self, idx):
        raise NotImplementedError(f"{self.name} does not support getitem 3D")
