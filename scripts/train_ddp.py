import os
from argparse import Namespace
from time import time
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
import torch.multiprocessing as mp
import torch.distributed as dist
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
    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f, root_path="exp")
    summary = DDPSummaryWriter(log_dir=recorder.tensorboard_path, rank=rank)
    # if the model is from the external package
    if cfg.MODEL.TYPE in EXT_PACKAGE:
        pkg = EXT_PACKAGE[cfg.MODEL.TYPE]
        exec(f"from lib.external import {pkg}")

    dist.barrier()  # wait for recoder to finish setup

    train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET, is_train=True)
    epoch_size = cfg.DATASET.TRAIN.EPOCH_SIZE
    # train_data = train_data.batched(arg.batch_size, collation_fn=collation_random_n_views)
    train_loader = wds.WebLoader(train_data,
                                 batch_size=arg.batch_size,
                                 num_workers=int(arg.workers),
                                 worker_init_fn=worker_init_fn,
                                 collate_fn=collation_random_n_views)
    train_loader = train_loader.with_epoch(epoch_size // arg.batch_size).shuffle(500)

    logger.warning(f"Using MixedWebDataset for training")

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

    optimizer = build_optimizer(model.parameters(), cfg=cfg.TRAIN)
    scheduler = build_scheduler(optimizer, cfg=cfg.TRAIN, steps=epoch_size // arg.batch_size * cfg.TRAIN.EPOCH)
    scheduler_type = cfg.TRAIN.SCHEDULER

    epoch = 0
    if arg.resume:
        epoch = recorder.resume_checkpoints(model, optimizer, scheduler, arg.resume, arg.resume_epoch)

    dist.barrier()  # wait for all processes to finish loading model
    logger.warning(f"############## start training from {epoch} to {cfg.TRAIN.EPOCH} ##############")
    for epoch_idx in range(epoch, cfg.TRAIN.EPOCH):
        model.train()
        trainbar = etqdm(train_loader, rank=rank, total=epoch_size // arg.batch_size)
        for bidx, batch in enumerate(trainbar):
            optimizer.zero_grad()
            step_idx = epoch_idx * (epoch_size // arg.batch_size) + bidx
            prd, loss_dict = model(batch, step_idx, "train", epoch_idx=epoch_idx)
            loss = loss_dict["loss"]
            loss.backward()
            if cfg.TRAIN.GRAD_CLIP_ENABLED:
                clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)

            optimizer.step()
            optimizer.zero_grad()
            trainbar.set_description(f"{bar_prefix['train']} Epoch {epoch_idx} "
                                     f"{model.module.format_metric('train')}")
            if scheduler_type == "CosineLR":
                scheduler.step()

        if scheduler_type != "CosineLR":
            scheduler.step()

        dist.barrier()  # wait for all processes to finish training
        logger.info(f"{bar_prefix['train']} Epoch {epoch_idx} | loss: {loss.item():.4f}, Done")
        logger.info(f"Current LR: {[group['lr'] for group in optimizer.param_groups]}")

        recorder.record_checkpoints(model, optimizer, scheduler, epoch_idx, arg.snapshot)
        torch.distributed.barrier()
        model.module.on_train_finished(recorder, epoch_idx)

        if (rank == 0  # only at rank 0,
                and epoch_idx != cfg.TRAIN.EPOCH - 1  # not the last epoch
                and epoch_idx % arg.eval_freq == 0):  # at eval freq, do validation
            logger.info("do validation and save results")
            with torch.no_grad():
                model.eval()
                valbar = etqdm(val_loader, rank=rank,
                               total=val_epoch_size // arg.val_batch_size)  # Only one gpu for validation
                for bidx, batch in enumerate(valbar):
                    step_idx = epoch_idx * (val_epoch_size // arg.val_batch_size) + bidx
                    model(batch, step_idx, "val", epoch_idx=epoch_idx)

            model.module.on_val_finished(recorder, epoch_idx)

    dist.destroy_process_group()
    # do last evaluation
    if rank == 0:
        logger.info("do last validation and save results")
        with torch.no_grad():
            model.eval()
            valbar = etqdm(val_loader, rank=rank, total=val_epoch_size // arg.val_batch_size)
            for bidx, batch in enumerate(valbar):
                step_idx = epoch_idx * (val_epoch_size // arg.val_batch_size) + bidx
                model(batch, step_idx, "val", epoch_idx=epoch_idx)

        model.module.on_val_finished(recorder, epoch_idx)


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
