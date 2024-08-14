import webdataset as wds
import braceexpand
import os
import torch
import numpy as np
from typing import List, Union
from torch import distributed as dist
from lib.utils.config import CN
from lib.utils.builder import build_transform
from ..utils.logger import logger
import random
import cv2

INV_EXTR_DATASETS = ['Interhand', 'Arctic', 'Oakink', 'Oakink2']


def expand(s):
    return os.path.expanduser(os.path.expandvars(s))


def expand_urls(urls: Union[str, List[str]]):
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand(url))]
    return urls


class MultiviewWebDataset():

    def __init__(self, cfg, data_preset=None, is_train=True):
        self.cfg = cfg
        self.data_split = cfg.DATA_SPLIT
        self.epoch_size = cfg.get("EPOCH_SIZE", None)
        self.data_preset = data_preset if data_preset is not None else cfg.DATA_PRESET
        self.urls = cfg.URLS
        self.name = cfg.URLS.split("/")[-1].split("_")[0]
        self.inv_extr = self.name in INV_EXTR_DATASETS
        self.random_n_views = cfg.get("RANDOM_N_VIEWS", False)
        self.view_range = cfg.get("VIEW_RANGE", None)
        self.mode = "train" if is_train else "val"
        self.transform = build_transform(cfg=cfg.TRANSFORM, data_preset=self.data_preset, is_train=is_train)

        if self.random_n_views:
            assert self.view_range is not None and self.view_range[0] >= 1

        dataset = wds.WebDataset(urls=expand_urls(self.urls),
                                 nodesplitter=wds.split_by_node,
                                 shardshuffle=True,
                                 resampled=False)  # resample should be set to false
        if is_train:
            dataset = dataset.shuffle(1000)

        dataset = dataset.decode('rgb8')
        dataset = dataset.map(self.process_data_item)
        self.dataset = dataset
        if self.epoch_size is not None:
            logger.info("Initialized MultiviewWebDataset {}_MV with epoch size {}.".format(self.name, self.epoch_size))
        else:
            logger.info("Initialized MultiviewWebDataset {}_MV for MixedWebDataset with mode {}.".format(
                self.name, self.mode))
        return None

    def process_data_item(self, item):
        n_view_imgs = {}
        for k in item.keys():
            if k.startswith("image"):
                img_type = "png" if "png" in k else "jpg"
                n_view_imgs[k] = item[k]

        n_cams = len(n_view_imgs)
        # dict of list
        # eg: key: [value_0, value_1, ...]
        key = item["__key__"]
        labels = item["label.pyd"]

        if "mano_pose" in labels:
            labels["mano_pose"] = [labels["mano_pose"][i].reshape(-1)[:48].reshape(16, 3) for i in range(n_cams)]
        else:
            # This is used as a temporary solution that deals with the case of Oakink dataset, should be fixed soon once the dataset is dumped properly
            labels["mano_pose"] = [np.zeros((16, 3)) for i in range(n_cams)]
            labels["mano_shape"] = [np.zeros(10) for _ in range(n_cams)]
        if self.inv_extr:
            labels['cam_extr'] = [np.linalg.inv(labels['cam_extr'][i]) for i in range(n_cams)]

        # random shuffle the camera idx
        indices = [i for i in range(0, n_cams)]
        if self.random_n_views:
            random.shuffle(indices)
            # randomly select from 1 to n ind from indices
            n = int(round(random.gauss(4, 2)))
            n = min(max(self.view_range[0], n), self.view_range[1])
            n = min(n, n_cams)  # in case the range is larger than the number of cameras
            indices_keep = indices[:n]
        else:
            indices_keep = indices

        new_master_id = indices_keep[0]
        new_master_serial = labels["cam_serial"][new_master_id]
        T_master_2_new_master = labels["cam_extr"][new_master_id]
        master_joints_3d = labels["joints_3d"][new_master_id]
        master_verts_3d = labels["verts_3d"][new_master_id]

        res = {}
        for ind in indices_keep:
            img = n_view_imgs[f"image_{ind}.{img_type}"]
            # print(labels["request_flip"])
            if labels.get("request_flip", False):
                cam_intr = labels["cam_intr"][ind]
                raw_size = labels["raw_size"][ind]
                cam_center = np.array([cam_intr[0, 2], cam_intr[1, 2]])
                M = np.array([[-1, 0, 2 * cam_center[0]], [0, 1, 0]], dtype=np.float32)
                # Use warpAffine to apply the reflection
                img = cv2.warpAffine(img, M, raw_size)

            lab = {k: v[ind] for k, v in labels.items() if k not in ["request_flip"]}
            # data aug
            tgt = self.transform(img, lab, no_rot=ind == new_master_id)

            # deal with camera extr
            T_m2c = lab["cam_extr"]
            T_new_master_2_cam = np.linalg.inv(T_master_2_new_master) @ T_m2c
            extr_prerot = tgt["extr_prerot"]  # (3, 3)
            extr_pre_transf = np.concatenate([extr_prerot, np.zeros((3, 1))], axis=1)
            extr_pre_transf = np.concatenate([extr_pre_transf, np.array([[0, 0, 0, 1]])], axis=0)
            T_new_master_2_cam = np.linalg.inv(extr_pre_transf @ np.linalg.inv(T_new_master_2_cam))
            tgt["target_cam_extr"] = T_new_master_2_cam.astype(np.float32)

            tgt.update(lab)
            for k, v in tgt.items():
                if k not in res:
                    res[k] = []
                res[k].append(v)

        for query in res.keys():
            if isinstance(res[query][0], (int, float, np.ndarray, torch.Tensor)):
                res[query] = np.stack(res[query])

        res["master_id"] = 0
        res["master_serial"] = new_master_serial
        res["master_joints_3d"] = master_joints_3d
        res["master_verts_3d"] = master_verts_3d
        res["__key__"] = key

        return res

    def get_dataset(self):
        return self.dataset
