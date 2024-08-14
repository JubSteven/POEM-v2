import numpy as np
from typing import Any, List, Dict
import torch
from ..utils.transform import batch_cam_extr_transf, batch_cam_intr_projection


def collation_random_n_views(batch):

    if not isinstance(batch, list):
        # Use list to align with the following format
        # This is the case where only 1 sample is provided.
        batch = [batch]

    batch_concat = dict()
    cam_view_number = [batch[i]["target_joints_3d"].shape[0] for i in range(len(batch))]

    for each in batch[0]:
        if isinstance(batch[0][each], np.ndarray) and not isinstance(batch[0][each][0], str):
            batch_concat[each] = np.concatenate([batch[i][each] for i in range(len(batch))], axis=0)
            batch_concat[each] = torch.Tensor(batch_concat[each])
        else:
            batch_concat[each] = [batch[i][each] for i in range(len(batch))]

    batch_concat["cam_view_num"] = np.array(cam_view_number)
    return batch_concat


# Records the common keys of all the multi-view datasets
def get_common_keys():
    return [
        'affine', 'target_joints_3d_no_rot', 'target_verts_3d_no_rot', 'rot_mat3d', 'target_bbox_scale',
        'target_verts_3d_rel', 'idx', 'verts_uvd', 'joints_vis', 'target_root_d', 'joints_3d', 'joints_2d',
        'master_joints_3d', 'target_bbox_center', 'target_cam_extr', 'joints_uvd', 'affine_postrot',
        'target_joints_uvd', 'rot_rad', 'target_verts_3d', 'target_joints_3d', 'master_id', 'target_cam_intr',
        'sample_idx', 'target_joints_2d', 'image', 'target_joints_vis', 'target_root_joint', 'bbox_scale',
        'extr_prerot', 'image_path', 'target_joints_3d_rel', 'target_verts_uvd', 'verts_3d', 'cam_center',
        'target_joints_heatmap', 'cam_intr', 'bbox_center', 'master_verts_3d', 'raw_size'
    ]


# Filter the keys of a sample to only include the common keys
def key_filter(sample):
    common_keys = get_common_keys()
    sample = {each: sample[each] for each in common_keys}
    return sample


def generate_grid_sample_proj(source_points, img_metas):
    target_points = []
    batch_size = len(img_metas["cam_view_num"])
    for i in range(batch_size):
        start_idx = np.sum(img_metas["cam_view_num"][:i])
        end_idx = np.sum(img_metas["cam_view_num"][:i + 1])

        # reshape bps_point to (1, N, nsample, 3)
        target_points_sub = source_points[i].repeat(1, img_metas["cam_view_num"][i], 1, 1)
        cam_extr_sub = img_metas["cam_extr"][start_idx:end_idx].unsqueeze(0)
        cam_intr_sub = img_metas["cam_intr"][start_idx:end_idx].unsqueeze(0)

        target_points_sub = batch_cam_extr_transf(torch.linalg.inv(cam_extr_sub), target_points_sub)
        target_points_sub = batch_cam_intr_projection(cam_intr_sub, target_points_sub)
        target_points.append(target_points_sub)

    target_points = torch.concat(target_points, dim=1).squeeze(0)
    return target_points
