from lib.utils.config import CN

from ..utils.builder import build_dataset
from .dexycb import DexYCB, DexYCBMultiView
from .ho3d import HO3D, HO3Dv3MultiView
from .oakink import OakInk
from .mix_dataset import MixDataset, MixWebDataset
from .interhand import InterHand
from .arctic import Arctic
from .freihand import FreiHAND_v2_Extra
from .yt3d import YT3D


def create_dataset(cfg: CN, data_preset: CN, is_train=True):
    """
    Create a dataset instance.
    """
    if cfg.TYPE == "MixDataset":
        # list of CN of each dataset
        if isinstance(cfg.DATASET_LIST, dict):
            dataset_list = [v for k, v in cfg.DATASET_LIST.items()]
        else:
            dataset_list = cfg.DATASET_LIST
            max_length = cfg.get("MAX_LENGTH", None)
        return MixDataset(cfg, dataset_list, max_length, data_preset)
    elif cfg.TYPE == "MixWebDataset":
        if isinstance(cfg.DATASET_LIST, dict):
            dataset_list = [v for k, v in cfg.DATASET_LIST.items()]
        else:
            dataset_list = cfg.DATASET_LIST
            max_length = cfg.get("MAX_LENGTH", None)
        return MixWebDataset(cfg, dataset_list, max_length, data_preset, is_train)
    else:
        # default building from cfg
        return build_dataset(cfg, data_preset=data_preset)
