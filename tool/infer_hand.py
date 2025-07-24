import os
import cv2

if __name__ == "__main__":
    # tune multi-threading params
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cv2.setNumThreads(0)

import random
import argparse
from argparse import Namespace
from time import time, sleep

import lib.models
from lib.models.model_abc import ModelABC
import numpy as np
import torch
from lib.external import EXT_PACKAGE
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.config import CN
import pickle
import json
from typing import Dict, Tuple, Optional

import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from lib.utils.transform import _affine_transform, _affine_transform_post_rot

from transform.transform_jit import inv_transf, transf_point_array
from transform.transform_np import inv_transf_np, transf_point_array_np, project_point_array_np

import ffmpeg
from video_tool.ffmpeg_util import FFMPEGFrameLoader

# use legacy viz context
from lib.viztools.viz_o3d_utils import VizContext

from .flip_util import flip_cam_extr


def bbox_get_center_scale(bbox, expand=2.0, mindim=200):
    w, h = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
    s = max(w, h)
    s = s * expand
    s = max(s, mindim)
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return np.array((center_x, center_y)), s


def format_batch(img_list, bbox_list, req_flip, camera_name_list, cam_intr_map, cam_extr_map, img_size, output_size, device):
    cam_serial_list, cam_intr_list, cam_extr_list, image_list = [], [], [], []
    for cam_name, img, bbox in zip(camera_name_list, img_list, bbox_list):
        if bbox is None:
            continue
        cam_intr_ori = cam_intr_map[cam_name]
        cam_extr_ori = cam_extr_map[cam_name]
        # get bbox center and bbox scale
        bbox_center, bbox_scale = bbox_get_center_scale(bbox)
        cam_center = np.array([cam_intr_ori[0, 2], cam_intr_ori[1, 2]])

        if req_flip:
            bbox_center[0] = 2 * cam_center[0] - bbox_center[0]
            # image & mask should be flipped horizontally with center at cam_center[0]
            # use cv2
            M = np.array([[-1, 0, 2 * cam_center[0]], [0, 1, 0]], dtype=np.float32)
            # Use warpAffine to apply the reflection
            img = cv2.warpAffine(img, M, img_size)
            cam_extr = inv_transf_np(flip_cam_extr(inv_transf_np(cam_extr_ori))).astype(np.float32)
        else:
            cam_extr = cam_extr_ori.copy()

        affine = _affine_transform(center=bbox_center, scale=bbox_scale, out_res=output_size, rot=0)
        affine_2x3 = affine[:2, :]
        imgcrop = cv2.warpAffine(img,
                                 affine_2x3, (int(output_size[0]), int(output_size[1])),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)

        cv2.imshow(cam_name, imgcrop[..., ::-1])

        image = tvF.to_tensor(imgcrop)
        assert image.shape[0] == 3
        image = tvF.normalize(image, [0.5, 0.5, 0.5], [1, 1, 1])

        cc = np.array([cam_intr_ori[0, 2], cam_intr_ori[1, 2]])
        affine_postrot = _affine_transform_post_rot(center=bbox_center,
                                                    scale=bbox_scale,
                                                    optical_center=cc,
                                                    out_res=output_size,
                                                    rot=0)
        cam_intr = affine_postrot.dot(cam_intr_ori)

        image_list.append(image)
        cam_serial_list.append(cam_name)
        cam_intr_list.append(cam_intr)
        cam_extr_list.append(cam_extr)
    
    cv2.waitKey(1)
    if len(cam_serial_list) <= 1:
        return None

    cam_view_num = np.array(len(cam_serial_list))
    cam_intr_th = torch.as_tensor(np.stack(cam_intr_list, axis=0)).to(device)
    cam_extr_th = torch.as_tensor(np.stack(cam_extr_list, axis=0)).to(device)
    cam_transf_th = inv_transf(cam_extr_th)
    image_th = torch.stack(image_list, axis=0).to(device)
    master_id = torch.as_tensor(0).to(device)
    master_serial = cam_serial_list[0]

    # modified cam_transf_th --> master should be identity
    master_cam_transf_th = cam_transf_th[0].unsqueeze(0)
    target_cam_trasnf_th = inv_transf(master_cam_transf_th) @ cam_transf_th

    batch = {
        "image": image_th,  # (n, 3, RES_X, RES_Y)
        "cam_serial": [cam_serial_list],
        "cam_view_num": cam_view_num[None],  # (1, )
        "target_cam_intr": cam_intr_th[None],  # (1, ?, 3, 3)
        "target_cam_extr": target_cam_trasnf_th[None],  # (1, ?, 4, 4)
        "master_id": master_id[None],  # (1, )
        "master_serial": [master_serial],
    }
    return batch


def extract_pred(pred, batch, req_flip, cam_extr_map):
    def flip_3d(annot_3d):
        annot_3d = annot_3d.copy()
        annot_3d[:, 0] = -annot_3d[:, 0]
        return annot_3d

    # for k, v in pred.items():
    #     print(k, v.shape)
    master_id = batch["master_id"][0]
    master_serial = batch["master_serial"][0]
    joint3d_in_master = pred["pred_joints_3d"][0]
    joint3d_in_master_np = joint3d_in_master.detach().cpu().numpy()
    vert3d_in_master = pred["pred_verts_3d"][0]
    vert3d_in_master_np = vert3d_in_master.detach().cpu().numpy()
    master_cam_extr = cam_extr_map[master_serial]
    if req_flip:
        master_cam_extr = inv_transf_np(flip_cam_extr(inv_transf_np(master_cam_extr)))
    master_cam_transf = inv_transf_np(master_cam_extr)
    joint3d_in_world_np = transf_point_array_np(master_cam_transf, joint3d_in_master_np)
    vert3d_in_world_np = transf_point_array_np(master_cam_transf, vert3d_in_master_np)
    if req_flip:
        joint3d_in_world_np = flip_3d(joint3d_in_world_np)
        vert3d_in_world_np = flip_3d(vert3d_in_world_np)
    return {
        "joints": joint3d_in_world_np,
        "verts": vert3d_in_world_np,
    }


CAMERA_INFO = {
    "011422072489": "camera_1",
    "818312071299": "camera_2",
    "050122071402": "camera_3",
}
VIDEO_SHAPE = (1280, 720)

DATA_FILEDIR = "/prefix/data/data"
MASK_FILEDIR = "/prefix/data/human_mask_hand"
CALIB_FILEDIR = "/prefix/data/calib/calib__2025_0319_1534_41"
HAND_SIDE_FILEPATH = "/prefix/data/hand_labels.json"


def main(
    cfg: CN,
    arg: Namespace,
    time_f: float,
):
    # load data
    print("load data")
    all_sequence_list = sorted(os.listdir(DATA_FILEDIR))

    print("poem-v2 start")

    # load model
    ## if the model is from the external package
    if cfg.MODEL.TYPE in EXT_PACKAGE:
        pkg = EXT_PACKAGE[cfg.MODEL.TYPE]
        exec(f"from lib.external import {pkg}")
    device = torch.device(f"cuda:0")
    model: ModelABC = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model.setup(summary_writer=None, log_freq=arg.log_freq)
    model.to(device)
    model.eval()

    hand_faces_np = model.face.detach().cpu().numpy()

    camera_name_list = list(CAMERA_INFO.values())

    # load param
    cam_extr_filedir = os.path.join(CALIB_FILEDIR, "cam_extr")
    cam_extr_map = {}
    for cam_name in camera_name_list:
        with open(os.path.join(cam_extr_filedir, f"{cam_name}.pkl"), "rb") as ifs:
            cam_extr_map[cam_name] = np.array(pickle.load(ifs), dtype=np.float32)
    cam_intr_filedir = os.path.join(CALIB_FILEDIR, "cam_intr")
    cam_intr_map = {}
    for cam_name in camera_name_list:
        with open(os.path.join(cam_intr_filedir, f"{cam_name}.pkl"), "rb") as ifs:
            cam_intr_map[cam_name] = np.array(pickle.load(ifs), dtype=np.float32)

    # load hand side
    hand_side_filepath = HAND_SIDE_FILEPATH
    with open(hand_side_filepath, "r") as ifs:
        hand_side_dict = json.load(ifs)
    hand_side_dict = {k: "rh" if v == "right" else "lh" for k, v in hand_side_dict.items()}

    # handle sequence
    # seq_curr = all_sequence_list[0]
    seq_curr = all_sequence_list[2]

    seq_filedir = os.path.join(DATA_FILEDIR, seq_curr)
    mask_filedir = os.path.join(MASK_FILEDIR, seq_curr)
    hand_side = hand_side_dict[seq_curr]

    process_seq(seq_filedir=seq_filedir,
                mask_filedir=mask_filedir,
                hand_side=hand_side,
                model=model,
                device=device,
                camera_name_list=camera_name_list,
                cam_extr_map=cam_extr_map,
                cam_intr_map=cam_intr_map,
                hand_faces_np=hand_faces_np)


def process_seq(seq_filedir, mask_filedir, hand_side, model, device, camera_name_list, cam_extr_map, cam_intr_map,
                hand_faces_np):
    loader_map = {}
    for cam_name in camera_name_list:
        loader_map[cam_name] = FFMPEGFrameLoader(
            os.path.join(seq_filedir, cam_name + ".mkv"),
            pix_fmt="rgb24",
            cache_size=256,
        )
    num_frame = loader_map[camera_name_list[0]].num_frame

    for frame_id in etqdm(range(num_frame), desc="processing", ncols=80):
        # load image
        img_list = []
        for cam_name in camera_name_list:
            img = loader_map[cam_name][frame_id]
            img_list.append(img)
        # load mask
        bboxes = []
        for cam_name in camera_name_list:
            mask = np.load(os.path.join(mask_filedir, cam_name, "bbox", f"{frame_id:05d}.npy"))
            mask = np.array(mask, dtype=np.float32)
            if mask.ndim > 1:
                mask = mask[0]
            bboxes.append(mask)
        if any([_bbox.shape[0] == 0 for _bbox in bboxes]):
            continue

        # process_right
        req_flip = hand_side == "lh"
        batch = format_batch(img_list=img_list,
                             bbox_list=bboxes,
                             req_flip=req_flip,
                             camera_name_list=camera_name_list,
                             cam_intr_map=cam_intr_map,
                             cam_extr_map=cam_extr_map,
                             img_size=VIDEO_SHAPE,
                             output_size=cfg.DATA_PRESET.IMAGE_SIZE,
                             device=device)
        with torch.no_grad():
            pred = model(batch, 0, "inference", epoch_idx=0)
        payload = extract_pred(pred, batch, req_flip=req_flip, cam_extr_map=cam_extr_map)

        # viz by reprojection
        hand_faces = hand_faces_np
        for cam_name, cam_extr in cam_extr_map.items():
            cam_intr = cam_intr_map[cam_name]

            _v3d = transf_point_array_np(cam_extr, payload["verts"])
            _v2d = project_point_array_np(cam_intr, _v3d)

            _img = img_list[camera_name_list.index(cam_name)].copy()
            _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)

            # paint 2d points
            for i in range(_v2d.shape[0]):
                _img = cv2.circle(_img, (int(_v2d[i, 0]), int(_v2d[i, 1])), 1, (0, 255, 0), cv2.FILLED)

            cv2.imshow("x", _img)
            while True:
                key = cv2.waitKey(1)
                if key == ord('\r'):
                    break


MODEL_CATEGORY = ['small', 'medium', 'large', 'huge', 'medium_MANO']
EMBED_SIZE = [128, 256, 512, 1024, 256]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # poem
    exp_time = time()
    arg, _ = parse_exp_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    # server
    main(cfg=cfg, arg=arg, time_f=exp_time)
