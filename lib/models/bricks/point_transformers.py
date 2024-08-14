import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from pytorch3d.ops import knn_points, sample_farthest_points
from lib.utils.points_utils import index_points
import os

def anchor_points(xyz, K, device):
    # TODO use fixed anchor each time.
    anchor_dir = os.path.join("assets", "anchor.npy")
    anchor_idx_dir = os.path.join("assets", "anchor_idx.npy")
    batch_size = xyz.shape[0]
    if not os.path.exists(anchor_dir):
        # the points of xyz are be the same set of BPS points loaded from bps.npy
        bps_points = xyz[0].unsqueeze(0) 
        local_xyz, local_idx = sample_farthest_points(bps_points, K=K) # batch_size = 1
        local_xyz_dump = local_xyz.cpu().detach().numpy()
        local_idx_dump = local_idx.cpu().detach().numpy()
        np.save(anchor_dir, local_xyz_dump)    
        np.save(anchor_idx_dir, local_idx_dump)
    else:
        local_xyz_load = np.load(anchor_dir)
        local_idx_load = np.load(anchor_idx_dir)
        local_xyz = torch.Tensor(local_xyz_load).to(device)
        local_idx = torch.Tensor(local_idx_load).to(device).type(torch.int64)
        
    local_xyz = local_xyz.repeat(batch_size, 1, 1)
    local_idx = local_idx.repeat(batch_size, 1)
        
    return local_xyz, local_idx


class ptTransition(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_channel, in_channel), nn.ReLU(), nn.Linear(in_channel, out_channel))

    def forward(self, points):
        new_points = self.fc(points)
        return new_points


# *  Modified to support Iterative Farthest Point Sampling
class ptTransformerBlock(nn.Module):

    def __init__(self, d_points, d_model, k, IFPS=False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.fc_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        self.IFPS = IFPS

    def forward(self, xyz, features):
        if self.training:
            x = cp.checkpoint(self._forward, xyz, features)
        else:
            x = self._forward(xyz, features)
        return x
            

    # xyz: b x n x 3, features: b x n x f
    def _forward(self, xyz, features):
        if self.IFPS:
            # Flawed here, further testing required
            # local_xyz := [4, 32, 3]
            # local_idx := [4, 32]
            local_xyz, local_idx = anchor_points(xyz, K=self.k, device=xyz.device)
            # Here the 'NN' for each point is the same
            # So we simply unsqueeze it to match it with knn case
            local_xyz = local_xyz.unsqueeze(1).repeat(1, 799, 1, 1)
            local_idx = local_idx.unsqueeze(1).repeat(1, 799, 1)            
        else:
            # local_idx := [4, 799, 32]
            # local_xyz := [4, 799, 32, 3]
            _, local_idx, local_xyz = knn_points(xyz, xyz, K=self.k, return_nn=True)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), local_idx), index_points(self.w_vs(x), local_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - local_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn


class ptTransformerBlock_CrossAttn(nn.Module):

    def __init__(self, d_points, d_model, k, expand_query_dim=False, IFPS=False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.fc_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.w_qs = nn.Linear(d_points, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        self.expand_query_dim = expand_query_dim
        self.IFPS = IFPS

        if expand_query_dim is True:
            self.fc_query = nn.Sequential(nn.Linear(d_points, d_points), nn.ReLU(), nn.Linear(d_points, d_points * 2))

    def forward(self, xyz, features, query):
        if self.training:
            x = cp.checkpoint(self._forward, xyz, features, query)
        else:
            x = self._forward(xyz, features, query)
        return x

    # xyz: b x n x 3, features: b x n x f, query = b x 799 x (3 + f)
    def _forward(self, xyz, features, query):
        query_xyz = query[:, :, :3]  # b x 799 x 3
        query_f = query[:, :, 3:]  # b x 799 x f

        if self.IFPS:
            local_xyz, local_idx = anchor_points(xyz, K=self.k, device=xyz.device)
            local_xyz = local_xyz.unsqueeze(1).repeat(1, 799, 1, 1)
            local_idx = local_idx.unsqueeze(1).repeat(1, 799, 1) 
        else:
            _, local_idx, local_xyz = knn_points(query_xyz, xyz, K=self.k, return_nn=True)
        # _, knn_idx, knn_xyz -> b x 799 x k, b x 799 x k, b x 799 x k x 3
        knn_features = index_points(features, local_idx)  # b x 799 x k x f

        pre = query_f  # b x 799 x f
        q = self.w_qs(query_f)  # b x 799 x d_model

        x = self.fc1(knn_features)  # b x 799 x k x d_model
        k = self.w_ks(x)  # k: b x 799 x k x d_model
        v = self.w_vs(x)  # v: b x 799 x k x d_model

        pos_enc = self.fc_delta(query_xyz[:, :, None] - local_xyz)  # b x 799 x k x d_model

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)  # b x 799 x 1 x d_model - b x 799 x k x d_model
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x 799 x k x d_model

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre  # b x 799 x f

        if self.expand_query_dim:
            res = self.fc_query(res)  # b x 799 x 2f

        return res, attn

