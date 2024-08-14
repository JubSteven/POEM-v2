import os
import random
from argparse import Namespace
from time import time

import lib.models
from lib.models.model_abc import ModelABC
import numpy as np
import torch
from lib.datasets import create_dataset
from lib.external import EXT_PACKAGE
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import CONST, bar_prefix, format_args_cfg
from lib.utils.net_utils import build_optimizer, build_scheduler, clip_gradient, setup_seed, worker_init_fn
from lib.utils.recorder import Recorder
from lib.utils.summary_writer import DDPSummaryWriter
from lib.utils.collation import collation_random_n_views
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from lib.utils.testing import IdleCallback, AUCCallback, DrawingHandCallback
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from lib.utils.config import CN
import webdataset as wds


def setup_ddp(arg, rank, world_size):
    """Setup distributed data parallel

    Args:
        arg (Namespace): arguments
        rank (int): rank of current process
        world_size (int): total number of processes, equal to number of GPUs
    """
    os.environ["MASTER_ADDR"] = arg.dist_master_addr
    os.environ["MASTER_PORT"] = arg.dist_master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    assert rank == torch.distributed.get_rank(), "Something wrong with DDP setup"
    torch.cuda.set_device(rank)
    dist.barrier()


def main_worker(rank: int, cfg: CN, arg: Namespace, world_size, time_f: float):
    setup_ddp(arg, rank, world_size)
    setup_seed(rank + cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f, root_path="exp", eval_only=True)
    summary = DDPSummaryWriter(log_dir=recorder.tensorboard_path, rank=rank)
    # if the model is from the external package
    if cfg.MODEL.TYPE in EXT_PACKAGE:
        pkg = EXT_PACKAGE[cfg.MODEL.TYPE]
        exec(f"from lib.external import {pkg}")

    dist.barrier()  # wait for recoder to finish setup

    if rank == 0:
        val_data = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET, is_train=False)
        val_epoch_size = cfg.DATASET.TEST.EPOCH_SIZE
        val_loader = wds.WebLoader(val_data,
                                   batch_size=arg.val_batch_size,
                                   num_workers=int(arg.workers),
                                   worker_init_fn=worker_init_fn,
                                   collate_fn=collation_random_n_views)
        

        val_loader = val_loader.with_epoch(val_epoch_size // arg.val_batch_size).shuffle(10)

        logger.warning(f"Using MixedWebDataset for validation")
    else:
        val_loader = None

    model: ModelABC = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model.setup(summary_writer=summary, log_freq=arg.log_freq)
    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=cfg.TRAIN.FIND_UNUSED_PARAMETERS, static_graph=True)

    dist.barrier()  # wait for all processes to finish loading model
    
    # define the callback, invoked after each batch forward
    if arg.eval_extra == "auc":
        val_max = cfg.TRAIN.get("VAL_MAX", 0.02)
        cb = AUCCallback(val_max=val_max, exp_dir=os.path.join(recorder.eval_dump_path))
    elif arg.eval_extra == "draw":
        cb = DrawingHandCallback(img_draw_dir=os.path.join(recorder.dump_path, "draws"))
    else:
        cb = IdleCallback()  # do nothing
    
    logger.warning(f"############## start validation ##############")
    with torch.no_grad():
        model.eval()
        valbar = etqdm(val_loader, rank=rank,
                        total=val_epoch_size // arg.val_batch_size)  # Only one gpu for validation
        for bidx, batch in enumerate(valbar):
            step_idx = 0 * (val_epoch_size // arg.val_batch_size) + bidx
            preds = model(batch, step_idx, "val", epoch_idx=0)
            cb(preds, batch, step_idx)

        model.module.on_val_finished(recorder, 0)
        cb.on_finished()

    dist.destroy_process_group()


if __name__ == "__main__":
    # tune multi-threading params
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    import cv2
    cv2.setNumThreads(0)

    exp_time = time()
    arg, _ = parse_exp_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id
    world_size = torch.cuda.device_count()

    if arg.resume:
        logger.warning(f"config will be reloaded from {os.path.join(arg.resume, 'dump_cfg.yaml')}")
        arg.cfg = os.path.join(arg.resume, "dump_cfg.yaml")
        cfg = get_config(config_file=arg.cfg, arg=arg)
    else:
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    logger.info("====> Use Distributed Data Parallel <====")
    mp.spawn(main_worker, args=(cfg, arg, world_size, exp_time), nprocs=world_size)
