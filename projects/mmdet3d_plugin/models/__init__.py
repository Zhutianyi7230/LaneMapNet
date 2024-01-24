# Copyright (c) OpenMMLab. All rights reserved.
from .builder import (build_fuser)###新增build_fuser

from .fusers import * ###new

__all__ = [
    'VOXEL_ENCODERS', 'MIDDLE_ENCODERS', 'FUSION_LAYERS', 'build_fuser'
]##新增'build_fuser'
