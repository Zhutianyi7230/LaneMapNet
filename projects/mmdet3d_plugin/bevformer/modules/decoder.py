# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import mmcv
import cv2 as cv
import copy
import warnings
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from projects.mmdet3d_plugin.datasets import bezier

import time
import numba
from numba  import cuda

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

# @cuda.jit
# def draw_line_on_grid(grids, bezier_points):
#     """grid #[bs*num_lines,100,200]
#         bezier_points #[bs*num_line,n_int,2]
#     """   
#     bs = grids.shape[0]
#     n_int = bezier_points.shape[1]
    
#     n = cuda.grid(1)
#     if n<bs*n_int and (n+1)%n_int!=0:
#         i = n // n_int
#         j = n % n_int
#         start = bezier_points[i][j]
#         end = bezier_points[i][j+1]
#         # Bresenham算法
#         dx = end[0] - start[0]
#         dy = end[1] - start[1]
#         x = start[0]
#         y = start[1]
#         sx = 1 if dx>0 else -1
#         sy = 1 if dy>0 else -1
#         dx = abs(dx)
#         dy = abs(dy)
        
#         if dx > dy:
#             err = dx / 2.0
#             while x != end[0]:
#                 grids[i, x, y] = True
#                 # index.append(int((y*W+x).cpu().numpy()))
#                 err -= dy
#                 if err < 0:
#                     y += sy
#                     err += dx
#                 x += sx
#         else:
#             err = dy / 2.0
#             while y != end[1]:
#                 grids[i, x, y] = True
#                 # index.append(int((y*W+x).cpu().numpy()))
#                 err -= dx
#                 if err < 0:
#                     x += sx
#                     err += dy
#                 y += sy
#         # 标记结束点
#         grids[i, end[0], end[1]] = True

#numba写法
@numba.jit(nopython=True,parallel=True)
def draw_line_on_grid(grid, start, end):
    """grid #[100,200]
       start, end#[2,]
    """   
    # Bresenham算法
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    x = start[0]
    y = start[1]
    sx = 1 if dx>0 else -1
    sy = 1 if dy>0 else -1
    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        err = dx / 2.0
        while x != end[0]:
            grid[x, y] = True
            # index.append(int((y*W+x).cpu().numpy()))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != end[1]:
            grid[x, y] = True
            # index.append(int((y*W+x).cpu().numpy()))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    # 标记结束点
    grid[end[0], end[1]] = True
    # index.append(int((end[1]*W+end[0]).cpu().numpy()))

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class CustomMSDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)  #torch.Size([1000, 1, 256])

        return self.dropout(output) + identity
    
@ATTENTION.register_module()
class BezierMSDeformableAttention(BaseModule):
    """An attention module bases on Deformable-Detr and BezierMSDeformableAttention.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_control_points=5,
                 n_int=10,
                 mode=0,# 0:simple deformable-att; 1:Bezier-ATT1 , 2:Bezier-ATT2
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.mode = mode
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_control_points = num_control_points
        self.n_int = n_int
        if self.mode==0:
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2)
            self.attention_weights = nn.Linear(embed_dims,
                                            num_heads * num_levels * num_points)
        elif self.mode==1 or self.mode==2:
            self.sampling_offsets_bezier1 = nn.Linear(
                embed_dims*2, num_heads * num_levels * num_points * 2)
            self.attention_weights_bezier1 = nn.Linear(embed_dims*2,
                                               num_heads * num_levels * num_points)
        
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()
        
        self.bezier = bezier.bezier_matrix(n_control=self.num_control_points,n_int=self.n_int)
        self.bezier = self.bezier.cuda()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        if self.mode==0:
            constant_init(self.sampling_offsets, 0.)
        elif self.mode==1 or self.mode==2:
            constant_init(self.sampling_offsets_bezier1, 0.)
        
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        if self.mode==0:
            self.sampling_offsets.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights, val=0., bias=0.)
        elif self.mode==1 or self.mode==2:
            self.sampling_offsets_bezier1.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights_bezier1, val=0., bias=0.)    

        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    # def draw_lines_on_grids(self, grid, ref_inter_points):
    #     """grid #[bs*100,100,200]
    #        ref_inter_points #[bs*100,5,2]
    #     """
    #     assert grid.shape[0]==ref_inter_points.shape[0]
    #     num_lines = grid.shape[0]
        
    #     indexes = []
    #     # 这一块可以用cuda并行
    #     for i in range(num_lines):
    #         index_sin_line = []
    #         for j in range(self.n_int-1):
    #             index_sin_line = index_sin_line + (self.draw_line_on_grid(grid[i],ref_inter_points[i,j],ref_inter_points[i,j+1]))
    #         indexes.append(index_sin_line)
        
    #     return indexes,grid
    
    #原版
    def draw_line_on_grid(self, grid, H, W, start, end):
        """grid #[100,200]
           start, end#[2,]
        """   
        index = []
        # Bresenham算法
        dx, dy = end - start
        x, y = start
        sx = torch.sign(dx)
        sy = torch.sign(dy)
        dx, dy = abs(dx), abs(dy)

        if dx > dy:
            err = dx / 2.0
            while x != end[0]:
                # grid[x, y] = True
                index.append(int((y*W+x).cpu().numpy()))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != end[1]:
                # grid[x, y] = True
                index.append(int((y*W+x).cpu().numpy()))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        # 标记结束点
        # grid[end[0], end[1]] = True
        index.append(int((end[1]*W+end[0]).cpu().numpy()))
        return index

            
    def get_bev_bezier_clue(self, bezier_points, bev_feat, H, W):
        '''
        从bev_feat中找出bezier_points对应像素的特征，并求平均.
        注意bezier_points中的坐标系和BEV坐标系不一样
        bezier_points #[bs,num_line,n_int,2]
        bev_feat #[bs,H*W,256]
        '''
        num_line = bezier_points.shape[1]
        index_x, index_y = bezier_points[:,:,:,0], H-1-bezier_points[:,:,:,1]
        index = index_y*W + index_x
        index = index.to(torch.long)
        bezier_feats = torch.gather(bev_feat.unsqueeze(1).expand([-1,num_line,-1,-1]),
                                    dim=2,
                                    index=index.unsqueeze(-1).expand(-1,-1,-1,256)) #torch.Size([1, 100, 10, 256])
        bezier_feats = bezier_feats.mean(2)        
        bezier_feats = bezier_feats.unsqueeze(2).expand([-1,-1,self.num_control_points,-1])#torch.Size([1, 100, num_control_points, 256])
        return bezier_feats

    def get_bev_bezier_clue_mode2(self, grids, bezier_points, bev_feat, H, W):
        '''
        根据bezier_points使用采样算法求出grids
        根据grids从bev_feat中取出对应像素位置的bev特征并求平均
        input:
        bezier_points #[bs*num_line,n_int,2]
        bev_feat #[bs,H*W,256]
        grids #[bs*num_line,W,H]
        output:
        bezier_feats #[bs*num_line, num_control_points, 256]
        '''
        bezier_points[:,:,1] = H-1-bezier_points[:,:,1]
        assert bezier_points.shape[0]==grids.shape[0]
        num_lines = bezier_points.shape[0]
        
        bezier_feats = torch.zeros([num_lines,self.num_control_points,self.embed_dims],device=bev_feat.device)#[bs*100,5,256]
        
        # #原写法
        # for i in range(num_lines):
        #     index_sin_line = []
        #     for j in range(self.n_int-1):
        #         index_sin_line = index_sin_line + (self.draw_line_on_grid(grids[i],H,W,bezier_points[i,j],bezier_points[i,j+1]))
            
        #     # bezier_feat = bev_feat[i//100,index_sin_line]
        #     bezier_feat = bev_feat[0,index_sin_line].mean(0)
        #     bezier_feats[i] = bezier_feat.unsqueeze(0).expand([self.num_control_points,-1])
        #     # indexes.append(index_sin_line)
        
        ##
        # 使用numba编译.
        grids = grids.detach().cpu().numpy()
        bezier_points = bezier_points.detach().cpu().numpy()
        for i in range(num_lines):
            # index_sin_line = []
            for j in range(self.n_int-1):
                # index_sin_line = index_sin_line + (draw_line_on_grid(grids[i], bezier_points[i,j],bezier_points[i,j+1]))
                draw_line_on_grid(grids[i], bezier_points[i,j],bezier_points[i,j+1])
            # bezier_feat = bev_feat[i//100,index_sin_line].mean(0)
            mask = torch.from_numpy(grids[i]).to(bev_feat.device)
            mask = mask.transpose(1,0).reshape([H*W,1])
            bezier_feat = torch.masked_select(bev_feat,mask).reshape([-1,self.embed_dims])#[n,256]
            bezier_feat = bezier_feat.mean(0).unsqueeze(0).expand([self.num_control_points,-1])
            bezier_feats[i] = bezier_feat
        ###
        
        # #cuda写法
        # grids = cuda.to_device(grids)
        # bezier_points = cuda.to_device(bezier_points)
        # threads_per_block = 256
        # blocks_per_grid = ((num_lines*self.n_int) + (threads_per_block - 1)) // threads_per_block
        # draw_line_on_grid[blocks_per_grid, threads_per_block](grids, bezier_points)
        # grids = grids.copy_to_host()
        # ###
        
        return bezier_feats



    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None, # #torch.Size([20000, bs, 256])
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None, # [bs, 100, 5, 1, 2]
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        
        num_line = reference_points.shape[1] #100
        num_control_points = reference_points.shape[2] #5

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert reference_points.shape[0]==bs
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)# (bs, num_query ,embed_dims)=(bs, 20000, 256)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        # # # ###BezierATT1
        if self.mode==1:
            reference_points = torch.reshape(reference_points,[bs, num_line, num_control_points, 2])# [bs, 100, 5, 2]
            ref_inter_points = torch.matmul(self.bezier.expand(reference_points.size(0),reference_points.size(1),-1,-1),reference_points)# [bs, 100, n_int, 2]
            factor = torch.tensor([spatial_shapes[0][1],spatial_shapes[0][0]],device=ref_inter_points.device)
            ref_inter_points = ref_inter_points * factor[None,None,None,:]
            ref_inter_points = ref_inter_points.int()
            
            bev_bezier = self.get_bev_bezier_clue(ref_inter_points, value, spatial_shapes[0][0], spatial_shapes[0][1])#torch.Size([1, 100, num_control_points, 256])
            bev_bezier = bev_bezier.reshape(bs,num_query,self.embed_dims)
            
            sampling_offsets = self.sampling_offsets_bezier1(torch.cat([query,bev_bezier],dim=-1)).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
            attention_weights = self.attention_weights_bezier1(torch.cat([query,bev_bezier],dim=-1)).view(
                bs, num_query, self.num_heads, self.num_levels * self.num_points)
            reference_points = reference_points.reshape([bs,num_line*num_control_points,2]).unsqueeze(2)

        
        # # ###BezierATT2
        elif self.mode==2:
            reference_points = torch.reshape(reference_points,[bs, num_line, num_control_points, 2])# [bs, 100, 5, 2]
            reference_points = torch.reshape(reference_points,[bs*num_line, reference_points.shape[2], reference_points.shape[3]])# [bs*100, 5, 2]
            ref_inter_points = torch.matmul(self.bezier.expand(reference_points.size(0),-1,-1),reference_points)# [bs*100, n_int, 2]
            factor = torch.tensor([spatial_shapes[0][1],spatial_shapes[0][0]],device=ref_inter_points.device)
            ref_inter_points = ref_inter_points * factor[None,None,:]
            ref_inter_points = ref_inter_points.int()
            
            H = spatial_shapes[0,0]
            W = spatial_shapes[0,1]        
            grids = torch.zeros((bs*num_line, W, H), dtype=torch.bool)#[bs*100,100,200]
            
            bev_bezier = self.get_bev_bezier_clue_mode2(grids, ref_inter_points, value, H, W)
            bev_bezier = bev_bezier.reshape(bs,num_query,self.embed_dims)

            #bezier1和bezier2用的是一样的linear层 
            sampling_offsets = self.sampling_offsets_bezier1(torch.cat([query,bev_bezier],dim=-1)).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
            attention_weights = self.attention_weights_bezier1(torch.cat([query,bev_bezier],dim=-1)).view(
                bs, num_query, self.num_heads, self.num_levels * self.num_points)
            ###
            reference_points = reference_points.reshape([bs,num_line*num_control_points,2]).unsqueeze(2)

        #origin
        elif self.mode==0:
            reference_points = reference_points.view(bs,-1,1,2)
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
            attention_weights = self.attention_weights(query).view(
                bs, num_query, self.num_heads, self.num_levels * self.num_points)
            ###
            
            attention_weights = attention_weights.softmax(-1)
            attention_weights = attention_weights.view(bs, num_query,
                                                    self.num_heads,
                                                    self.num_levels,
                                                    self.num_points)
            reference_points = reference_points.reshape([bs,num_line*num_control_points,1,2])
        
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        value = value.view(bs, num_value, self.num_heads, -1)# (bs, 20000, 8, 32)        
        if torch.cuda.is_available() and value.is_cuda:
            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)  #torch.Size([5*100, 1, 256])
        return self.dropout(output) + identity
