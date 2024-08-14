import cv2
import numpy as np
import torch
import torch.nn as nn

from ..utils.builder import MODEL
from ..utils.logger import logger
from .backbones import build_backbone
from ..utils.recorder import Recorder
from ..utils.net_utils import load_weights
from lib.metrics import LossMetric, MeanEPE
from lib.metrics.basic_metric import AverageMeter
from ..utils.misc import param_size
from .model_abc import ModelABC
from ..viztools.draw import draw_batch_joint_images
from lib.utils.heatmap import get_heatmap_pred, accuracy_heatmap


def tensor2array(tensor, max_value=None, colormap='jet', channel_first=True, mean=0.5, std=0.5):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            color_cvt = cv2.COLOR_BGR2RGB
            if colormap == 'jet':
                colormap = cv2.COLORMAP_JET
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255 * tensor.squeeze().numpy() / max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32) / 255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy() / max_value).clip(0, 1)
        if channel_first:
            array = array.transpose(2, 0, 1)
    elif tensor.ndimension() == 3:
        assert (tensor.size(0) == 3)
        array = ((mean + tensor.numpy() * std) * 255).astype(np.uint8)
        if not channel_first:
            array = array.transpose(1, 2, 0)

    return array


class JointsMSELoss(nn.Module):

    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


@MODEL.register_module()
class DarkPose_ResNet(ModelABC):

    def __init__(self, cfg):
        super(DarkPose_ResNet, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.data_preset_cfg = cfg.DATA_PRESET
        self.num_joints = cfg.DATA_PRESET.NUM_JOINTS
        self.img_backbone = build_backbone(cfg.BACKBONE, data_preset=self.data_preset_cfg)

        self.inplanes = cfg.INPLANES
        extra = cfg.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(in_channels=extra.NUM_DECONV_FILTERS[-1],
                                     out_channels=cfg.NUM_JOINTS,
                                     kernel_size=extra.FINAL_CONV_KERNEL,
                                     stride=1,
                                     padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        self.criterion = JointsMSELoss()
        self.MPJPE = MeanEPE(cfg, "Joints_2D")
        self.loss_metric = LossMetric(cfg)
        self.heatmap_acc = AverageMeter(name='heatmap_acc')

        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL
        load_weights(self, pretrained=cfg.PRETRAINED)
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def setup(self, summary_writer, **kwargs):
        self.summary = summary_writer

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(in_channels=self.inplanes,
                                   out_channels=planes,
                                   kernel_size=kernel,
                                   stride=2,
                                   padding=padding,
                                   output_padding=output_padding,
                                   bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            # layers.append(nn.MaxPool2d(2))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def extract_img_feat(self, img):
        B = img.size(0)
        if img.dim() == 5:
            if img.size(0) == 1 and img.size(1) != 1:  # (1, N, C, H, W)
                img = img.squeeze(0)  # (N, C, H, W)
            else:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
        
        img_feats = self.img_backbone(image=img)
        if isinstance(img_feats, dict):
            img_feats = list([v for v in img_feats.values() if len(v.size()) == 4])

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))  # (B, N, C, H, W)

        return img_feats_reshaped

    def _forward_impl(self, batch, **kwargs):
        preds = {}
        img = batch["image"]  # (B, 3, H, W)
        img_feats = self.extract_img_feat(img)  # [(B, N, C, H, W), ...]
        x = self.deconv_layers(img_feats[-1].squeeze(1))  # [B, 256, 64, 64]
        preds['pred_heatmap'] = self.final_layer(x)  # [B, 21, 64, 64]
        return preds

    def compute_loss(self, preds, gt):
        loss_dict = {}
        target = gt['target_joints_heatmap']  # [B, 21, 128, 128]
        loss = self.criterion(preds['pred_heatmap'], target)
        loss_dict['loss'] = loss
        return loss, loss_dict

    def batch_multi2single(self, batch):
        if batch["image"].dim() == 5:
            batch["image"] = batch["image"].flatten(0, 1)
            batch["target_joints_heatmap"] = batch["target_joints_heatmap"].flatten(0, 1)
            batch["target_joints_2d"] = batch["target_joints_2d"].flatten(0, 1)
        if batch["image"].dim() == 3:
            batch["image"] = batch["image"].unsqueeze(0)
            batch["target_joints_heatmap"] = batch["target_joints_heatmap"].unsqueeze(0)
            batch["target_joints_2d"] = batch["target_joints_2d"].unsqueeze(0)
        return batch

    def training_step(self, batch, step_idx, **kwargs):
        batch = self.batch_multi2single(batch)
        batch_size = batch["image"].shape[0]
        preds = self._forward_impl(batch, **kwargs)
        loss, loss_dict = self.compute_loss(preds, batch)

        self.loss_metric.feed(loss_dict, batch_size)
        if step_idx % self.train_log_interval == 0:
            self.summary.add_scalar("loss", loss.item(), step_idx)

        viz_interval = self.train_log_interval * 5
        if step_idx % viz_interval == 0:
            self.board_img(
                'train',
                step_idx,
                batch['image'][0],
                htmp_gt=batch.get('target_joints_heatmap'),
                htmp_pred=preds.get('pred_heatmap'),
            )
            img_toshow = batch["image"]
            uv_coord, _ = get_heatmap_pred(preds['pred_heatmap'])  # [B, 21, 2]
            W, H = img_toshow.shape[-1], img_toshow.shape[-2]
            f = preds['pred_heatmap'].shape[-1]
            uv_coord_toshow = torch.einsum("bij, j->bij", uv_coord, torch.tensor([W / f, H / f]).to(uv_coord.device))
            gt_uv = batch["target_joints_2d"]
            img_array = draw_batch_joint_images(uv_coord_toshow, gt_uv, img_toshow, step_idx)
            self.summary.add_image(f"img/train_uv_val", img_array, step_idx, dataformats="NHWC")

        return preds, loss_dict

    def board_img(self, phase, n_iter, img, **kwargs):
        self.summary.add_image(phase + '/img', tensor2array(img), n_iter)
        if kwargs.get('htmp_pred') is not None:
            self.summary.add_image(phase + '/htmp_gt', tensor2array(kwargs['htmp_gt'][0].sum(dim=0).clamp(max=1)),
                                   n_iter)
            self.summary.add_image(phase + '/htmp_pred', tensor2array(kwargs['htmp_pred'][0].sum(dim=0).clamp(max=1)),
                                   n_iter)

    def on_train_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train-"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        self.loss_metric.reset()

    def validation_step(self, batch, step_idx, **kwargs):
        batch = self.batch_multi2single(batch)
        preds = self.testing_step(batch, step_idx, **kwargs)
        # uv_coord = preds['pred_uv_coord']
        uv_coord_toshow = preds['pred_uv_coord_toshow']
        gt_uv = batch["target_joints_2d"]

        self.summary.add_scalar("MPJPE", self.MPJPE.get_result(), step_idx)
        self.summary.add_scalar("heatmap_acc", self.heatmap_acc.avg, step_idx)

        self.board_img(
            'val',
            step_idx,
            batch['image'][0],
            htmp_gt=batch.get('target_joints_heatmap'),
            htmp_pred=preds.get('pred_heatmap'),
        )
        if step_idx % (self.train_log_interval * 10) == 0:
            img_toshow = batch["image"]
            img_array = draw_batch_joint_images(uv_coord_toshow, gt_uv, img_toshow, step_idx)
            self.summary.add_image(f"img/train_uv_val", img_array, step_idx, dataformats="NHWC")

    def on_val_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val-"
        recorder.record_metric([self.MPJPE, self.heatmap_acc], epoch_idx, comment=comment)
        self.MPJPE.reset()
        self.heatmap_acc.reset()

    def testing_step(self, batch, step_idx, **kwargs):
        batch = self.batch_multi2single(batch)
        img = batch["image"]  # (BN, 3, H, W) 4 dimensions
        inp_img_shape = img.shape[-2:]  # H, W
        H, W = inp_img_shape

        preds = self._forward_impl(batch)
        pred_heatmap = preds["pred_heatmap"]  # [B, 21, 64, 64]

        uv_coord, _ = get_heatmap_pred(pred_heatmap)  # [B, 21, 2] rang: 0 ~ 64

        W, H = img.shape[-1], img.shape[-2]
        f = preds['pred_heatmap'].shape[-1]
        uv_coord_toshow = torch.einsum("bij, j->bij", uv_coord, torch.tensor([W / f, H / f]).to(uv_coord.device))

        gt_uv = batch["target_joints_2d"]
        self.MPJPE.feed(uv_coord_toshow, gt_uv)

        gt_heatmap = batch['target_joints_heatmap']
        BATCH_SIZE = gt_heatmap.shape[0]
        device = gt_heatmap.device
        kp_vis = torch.ones([BATCH_SIZE, self.num_joints]).to(device)
        acc, _ = accuracy_heatmap(preds['pred_heatmap'], gt_heatmap, kp_vis)
        self.heatmap_acc.update_by_mean(acc, BATCH_SIZE)

        if step_idx % self.train_log_interval == 0:
            self.board_img(
                'test',
                step_idx,
                batch['image'][0],
                htmp_gt=batch.get('target_joints_heatmap'),
                htmp_pred=preds.get('pred_heatmap'),
            )
            img_toshow = batch["image"]
            img_array = draw_batch_joint_images(uv_coord_toshow, gt_uv, img_toshow, step_idx)
            self.summary.add_image(f"img/test_uv_val", img_array, step_idx, dataformats="NHWC")

        preds['pred_uv_coord'] = uv_coord
        preds['pred_uv_coord_toshow'] = uv_coord_toshow

        if "callback" in kwargs:
            callback = kwargs.pop("callback")
            callback(preds, batch, step_idx, **kwargs)

        return preds

    def on_test_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-test-"
        recorder.record_metric([self.MPJPE, self.heatmap_acc], epoch_idx, comment=comment)
        self.MPJPE.reset()
        self.heatmap_acc.reset()

    def format_metric(self, mode="train"):
        if mode == "val":
            metric_toshow = [self.MPJPE, self.heatmap_acc]
            return " | ".join([str(me) for me in metric_toshow])
        elif mode == "test":
            metric_toshow = [self.MPJPE, self.heatmap_acc]
            return " | ".join([str(me) for me in metric_toshow])
        elif mode == "train":
            return f"{self.loss_metric.get_loss('loss'):.4f}"
        else:
            return ""

    def forward(self, inputs, step_idx, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx, **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, step_idx, **kwargs)
        elif mode == "test":
            return self.testing_step(inputs, step_idx, **kwargs)
        elif mode == "inference":
            return self.inference_step(inputs, step_idx, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")
