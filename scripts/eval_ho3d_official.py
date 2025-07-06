import argparse
from argparse import Namespace
import yaml
import os
import subprocess
from lib.utils.config import CN
from lib.datasets.ho3d_official_test import HO3DOfficialTestMultiView
import torch

from torch.utils.data import DataLoader
from lib.models.model_abc import ModelABC
from lib.utils import builder
from lib.utils.recorder import Recorder
from lib.utils.summary_writer import DDPSummaryWriter
from time import time
from lib.utils.etqdm import etqdm
from lib.utils.collation import collation_random_n_views
from lib.utils.config import get_config
from lib.utils.testing import IdleCallback, AUCCallback, DrawingHandCallback, HO3DOfficialEvalCallback
from lib.utils.logger import logger

UNDEFINED = None

HO3D_OFFICIAL_TEST_CFG = dict(
    DATA_MODE="3D",
    VERSION=UNDEFINED,
    DATA_ROOT="data",
    DATA_SPLIT="test",
    N_VIEWS=5,
    RANDOM_N_VIEWS=False,
    SPLIT_MODE="paper",
    ADD_EVALSET_TRAIN=True,
    FILTER_INVISIBLE_HAND=True,
    MASTER_SYSTEM="as_first_camera",
    TRANSFORM=dict(
        TYPE="SimpleTransform3DMultiView",
        AUG=False,
        CENTER_JIT=0.05,
        SCALE_JIT=0.06,
        COLOR_JIT=0.3,
        ROT_JIT=10,
        ROT_PROB=1.0,
        OCCLUSION=False,
        OCCLUSION_PROB=0.0,
    ),
    DATA_PRESET=dict(
        USE_CACHE=True,
        BBOX_EXPAND_RATIO=2.0,
        IMAGE_SIZE=(256, 256),
        CENTER_IDX=0,
    ),
)

MODEL_CATEGORY = ['small', 'medium', 'large', 'huge', 'medium_MANO']
EMBED_SIZE = [128, 256, 512, 1024, 256]


def main(cfg: CN, arg: Namespace, time_f: float):

    cfg_dataset = CN(HO3D_OFFICIAL_TEST_CFG)
    if arg.ho3d_v == "2":
        cfg_dataset.VERSION = "v2"
    elif arg.ho3d_v == "3":
        cfg_dataset.VERSION = "v3"
    else:
        raise ValueError(f"Unsupported HO3D version: {arg.ho3d_v}")

    dataset = HO3DOfficialTestMultiView(cfg_dataset)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=collation_random_n_views)

    # set the embed size based on the model type
    model_idx = MODEL_CATEGORY.index(arg.model)
    embed_size = EMBED_SIZE[model_idx]
    print(embed_size)
    cfg["MODEL"]["HEAD"]["POSITIONAL_ENCODING"]["NUM_FEATS"] = embed_size // 2
    cfg["MODEL"]["HEAD"]["TRANSFORMER"]["INPUT_FEAT_DIM"] = embed_size
    cfg["MODEL"]["HEAD"]["POINTS_FEAT_DIM"] = embed_size
    cfg["MODEL"]["HEAD"]["EMBED_DIMS"] = embed_size

    # Set the parametric output for medium_MANO
    if arg.model == "medium_MANO":
        cfg["MODEL"]["HEAD"]["TRANSFORMER"]["PARAMETRIC_OUTPUT"] = True
    else:
        cfg["MODEL"]["HEAD"]["TRANSFORMER"]["PARAMETRIC_OUTPUT"] = False

    exp_time = time()
    rank = 0
    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=exp_time, root_path="exp", eval_only=True)
    summary = DDPSummaryWriter(log_dir=recorder.tensorboard_path, rank=rank)
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model.setup(summary_writer=summary, log_freq=arg.log_freq)
    model = model.to(device=rank)

    # define the callback, invoked after each batch forward
    if arg.eval_extra == "auc":
        val_max = cfg.TRAIN.get("VAL_MAX", 0.02)
        cb = AUCCallback(val_max=val_max, exp_dir=os.path.join(recorder.eval_dump_path))
    elif arg.eval_extra == "draw":
        cb = DrawingHandCallback(img_draw_dir=os.path.join(recorder.dump_path, "draws"))
    elif arg.eval_extra == "ho3d_offi":
        cb = HO3DOfficialEvalCallback(exp_dir=os.path.join(recorder.eval_dump_path))
    else:
        cb = IdleCallback()

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(etqdm(dataloader)):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(rank)

            _ = model(batch, i, "val", epoch_idx=0, callback=cb)

        model.on_val_finished(recorder, 0)
        cb.on_finished()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Single Setting")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--model", type=str, required=True, help="Model category.")
    parser.add_argument("--gpu_id", "-g", type=int, default=0, required=True, help="GPU ID to run the evaluation.")
    parser.add_argument("--reload", type=str, default=None, help="Path to the checkpoint to reload.")
    parser.add_argument("--port", "-p", type=int, default=60000, help="Port to run the evaluation.")
    parser.add_argument("--exp_id", default="default", type=str, help="Experiment ID")
    parser.add_argument("--log_freq", default=10, type=int, help="How often to write summary logs")
    parser.add_argument("--ho3d-v", default="3", type=str, choices=["3", "2"], help="HO3D version to use, 2 or 3")
    parser.add_argument("--eval_extra",
                        default="none",
                        type=str,
                        choices=["none", "auc", "draw", "ho3d_offi"],
                        help="Extra mode for testing, e.g. `draw`: test with drawing")

    exp_time = time()
    arg = parser.parse_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(arg.cfg)
    if arg.reload is not None:
        logger.warning(f"cfg MODEL's pretrained {cfg.MODEL.PRETRAINED} reset to arg.reload: {arg.reload}")
        cfg.MODEL.PRETRAINED = arg.reload

    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    print(world_size)

    main(cfg=cfg, arg=arg, time_f=exp_time)
