# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                                  ROI_EXTRACTORS, SHARED_HEADS)

FUSERS = Registry('fusers')###new

def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)

def build_fuser(cfg):
    """build fusion layer between image and pointcloud."""
    return FUSERS.build(cfg)


