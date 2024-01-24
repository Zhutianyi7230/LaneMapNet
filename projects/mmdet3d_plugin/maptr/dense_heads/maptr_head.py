import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.utils import TORCH_VERSION, digit_version
from .matcher import build_matcher ###
from .postprocess import build_postprocess ###
from .utils import get_vertices,get_merged_coeffs,my_color_line_maker
from projects.mmdet3d_plugin.datasets import bezier

def curvature_loss(points):
    # Ensure points is a torch tensor
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)

    # Compute first derivatives
    dx = (points[2:, 0] - points[:-2, 0]) / 2
    dy = (points[2:, 1] - points[:-2, 1]) / 2

    # Compute second derivatives
    ddx = points[2:, 0] - 2 * points[1:-1, 0] + points[:-2, 0]
    ddy = points[2:, 1] - 2 * points[1:-1, 1] + points[:-2, 1]

    # Compute curvature
    curvature = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5)
    d_curvature = curvature[1:] - curvature[:-1]
    loss = torch.mean(d_curvature**2)
    return loss


def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h,patch_w,patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])

    return bboxes
def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@HEADS.register_module()
class MapTRHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 num_vec=20,
                 num_pts_per_vec=2,
                 num_pts_per_gt_vec=2,
                 num_line_vec=100,###
                 num_control_points=5,###
                 num_stsu_classes=2,###
                 stsu_losses=['labels', 'boxes','loss_polyline', 'endpoints','assoc'],###
                 query_embed_type='all_pts',
                 transform_method='minmax',
                 gt_shift_pts_pattern='v0',
                 dir_interval=1,
                 loss_pts=dict(type='ChamferDistance', 
                             loss_src_weight=1.0, 
                             loss_dst_weight=1.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.num_line_vec = num_line_vec###
        self.num_control_points = num_control_points###3
        self.num_coeffs = num_control_points*2###3*2=6
        self.num_stsu_classes = num_stsu_classes ###2
        self.stsu_losses = stsu_losses###
        
        self.dir_interval = dir_interval

        super(MapTRHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.loss_pts = build_loss(loss_pts)
        self.loss_dir = build_loss(loss_dir)
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.num_line_vec = num_line_vec###
        self.num_control_points = num_control_points###3
        self.num_coeffs = num_control_points*2###3*2=6
        self.num_stsu_classes = num_stsu_classes ###
        self.stsu_losses = stsu_losses###
        self.matcher = build_matcher()###
        self.postprocess = build_postprocess()###
        self._init_layers()
        
        self.bezierA = bezier.bezier_matrix(n_control=self.num_control_points,n_int=50)
        self.bezierA = self.bezierA.cuda()
        self.bezierB = bezier.bezier_matrix(n_control=self.num_control_points,n_int=20)
        self.bezierB = self.bezierB.cuda()
        self.bezierC = bezier.bezier_matrix(n_control=self.num_control_points,n_int=5)
        self.bezierC = self.bezierC.cuda()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        # cls_branch.append(Linear(self.embed_dims * 2, self.embed_dims))
        # cls_branch.append(nn.LayerNorm(self.embed_dims))
        # cls_branch.append(nn.ReLU(inplace=True))
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        
        stsu_cls_branch = []
        for _ in range(self.num_reg_fcs):#2
            stsu_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            stsu_cls_branch.append(nn.ReLU())
        stsu_cls_branch.append(Linear(self.embed_dims, 3))#0 for None, 1 for line_center, 2 for line_connector
        stsu_cls_branch = nn.Sequential(*stsu_cls_branch)       
        
        stsu_reg_branch = []
        for _ in range(self.num_reg_fcs):#2
            stsu_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            stsu_reg_branch.append(nn.ReLU())
        stsu_reg_branch.append(Linear(self.embed_dims, self.code_size))
        stsu_reg_branch = nn.Sequential(*stsu_reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers
        num_stsu_layers = (self.transformer.line_decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.line_decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.stsu_reg_branches = _get_clones(stsu_reg_branch, num_stsu_layers)
            self.stsu_cls_branches = _get_clones(stsu_cls_branch, num_stsu_layers)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.stsu_reg_branches = nn.ModuleList(
                [stsu_reg_branch for _ in range(num_stsu_layers)])
            self.stsu_cls_branches = nn.ModuleList(
                [stsu_cls_branch for _ in range(num_stsu_layers)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            
            self.line_instance_embedding = nn.Embedding(self.num_line_vec,self.embed_dims * 2)###(100,512)
            self.line_pts_embedding = nn.Embedding(self.num_control_points,self.embed_dims * 2)###(5,512)
            
            if self.query_embed_type == 'all_pts':
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
            elif self.query_embed_type == 'instance_pts':
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)
        
        self.association_embed_maker = MLP(self.embed_dims, 256, 128, 3)
        self.association_classifier = MLP(2*128,256,1,3)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], 0.)
    
    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, prev_ref_pts=None,  only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        # import pdb;pdb.set_trace()
        if self.query_embed_type == 'all_pts':
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == 'instance_pts':
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        
        line_instance_embeds = self.line_instance_embedding.weight.unsqueeze(1)
        line_pts_embeds = self.line_pts_embedding.weight.unsqueeze(0)
        line_query_embeds = (line_pts_embeds + line_instance_embeds).flatten(0, 1).to(dtype)   
           
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                line_query_embeds,###
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                stsu_reg_branches=self.stsu_reg_branches, ###
                img_metas=img_metas,
                prev_bev=prev_bev,
                prev_ref_pts=prev_ref_pts
        )

        bs = bs//2
        
        bev_embed, hs, init_reference, inter_references,\
            hs_line, init_reference_line, inter_references_line  = outputs
        #hs.shape = torch.Size([6, 1, 1000, 256])
            
        hs = hs.permute(0, 2, 1, 3) #torch.Size([6, 1, 1000, 256])
        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # import pdb;pdb.set_trace()
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # vec_embedding = hs[lvl].reshape(bs, self.num_vec, -1)
            outputs_class = self.cls_branches[lvl](hs[lvl]
                                            .view(bs,self.num_vec, self.num_pts_per_vec,-1)
                                            .mean(2))
            
            tmp = self.reg_branches[lvl](hs[lvl])#预测的坐标是相对于该层初始ref points的偏移

            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            if lvl==(hs.shape[0]-1) :
                last_reference = tmp
            # tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp = tmp.sigmoid() # cx,cy,w,h
            # import pdb;pdb.set_trace()
            # tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
            #                  self.pc_range[0]) + self.pc_range[0])
            # tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
            #                  self.pc_range[1]) + self.pc_range[1])
            # tmp = tmp.reshape(bs, self.num_vec,-1)
            # TODO: check if using sigmoid
            outputs_coord, outputs_pts_coord = self.transform_box(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_pts_preds': outputs_pts_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'enc_pts_preds': None,
            'last_reference':last_reference#这个是没用的，懒得删了
        }
        
        
        ###
        hs_line = hs_line.permute(0, 2, 1, 3) #torch.Size([4, 1, 100*5, 256])
        outputs_line_classes = []
        outputs_line_pts_coords = []
        assoc_features_line = []
        for lvl in range(hs_line.shape[0]):
            if lvl == 0:
                reference = init_reference_line
            else:
                reference = inter_references_line[lvl - 1]
            outputs_line_class = self.stsu_cls_branches[lvl](hs_line[lvl]
                                            .view(bs,self.num_line_vec, self.num_control_points,-1)
                                            .mean(2))
        
            reference = inverse_sigmoid(reference)
            tmp = self.stsu_reg_branches[lvl](hs_line[lvl])#预测的坐标是相对于该层初始ref points的偏移

            assert reference.shape[-1] == 2
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            
            assoc_feature_line = self.association_embed_maker(hs_line[lvl]
                                                                   .view(bs,self.num_line_vec, self.num_control_points,-1)
                                                                   .mean(2)) #[bs, 100, 128]
            tmp = tmp.sigmoid() 
            
            outputs_line_pts_coord = tmp.view(tmp.shape[0], self.num_line_vec, self.num_control_points*2)
            
            outputs_line_classes.append(outputs_line_class)
            outputs_line_pts_coords.append(outputs_line_pts_coord)
            assoc_features_line.append(assoc_feature_line)
            
        outputs_line_classes = torch.stack(outputs_line_classes) # [4, bs, 100, 3]
        outputs_line_pts_coords = torch.stack(outputs_line_pts_coords) # [4, bs, 100, 5*2]
        assoc_features_line = torch.stack(assoc_features_line)# [4, bs, 100, 128]
        
        outs['pred_logits'] = outputs_line_classes 
        outs['pred_spline'] = outputs_line_pts_coords
        outs['assoc_features'] = assoc_features_line

        return outs
    

    def transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.num_vec,
                                self.num_pts_per_vec,2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # import pdb;pdb.set_trace()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        # import pdb;pdb.set_trace()
        assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)

        
        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # import pdb;pdb.set_trace()
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def get_assoc_estimates(self,  outputs, indices):
            
        _, idx = self._get_src_permutation_idx(indices)
        #idx= tensor([ 3,  6, 10, 13, 15, 22, 28, 30, 32, 36, 50, 58, 64, 65, 71, 87])
#        _, target_ids = self._get_tgt_permutation_idx(indices)

        assoc_features = torch.squeeze(outputs['assoc_features'])
        
        selected_features = assoc_features[idx]#torch.Size([16, 128]) 从100个里选出与GT配对的16个
        
        reshaped_features1 = torch.unsqueeze(selected_features,dim=1).repeat(1,selected_features.size(0),1)
        reshaped_features2 = torch.unsqueeze(selected_features,dim=0).repeat(selected_features.size(0),1,1)
        
        total_features = torch.cat([reshaped_features1,reshaped_features2],dim=-1)#torch.Size([16, 16, 256])

        est = torch.squeeze(self.association_classifier(total_features),dim=-1)#torch.Size([16, 16])

        outputs['pred_assoc'] = torch.unsqueeze(est,dim=0)
        
        return outputs

    def thresh_and_assoc_estimates(self,outputs,thresh=0.5):

        assoc_features = torch.squeeze(outputs['assoc_features'])#[100,128]
        
        out_logits = torch.squeeze(outputs['pred_logits'])#[100,3]
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        selected_mask = (labels!=0) * (scores>thresh)

        selected_features = assoc_features[selected_mask]#只挑分数大于static_thresh的参与构建assoc矩阵
        
        reshaped_features1 = torch.unsqueeze(selected_features,dim=1).repeat(1,selected_features.size(0),1)
        reshaped_features2 = torch.unsqueeze(selected_features,dim=0).repeat(selected_features.size(0),1,1)
        
        total_features = torch.cat([reshaped_features1,reshaped_features2],dim=-1)
        
        est = torch.squeeze(self.association_classifier(total_features).sigmoid(),dim=-1)
        
        outputs['pred_assoc'] = torch.unsqueeze(est,dim=0)

        return outputs #在训练刚开始的时候outputs.shape =torch.Size([1, 0, 0]),因为没有>0.5的分数

    def get_selected_estimates(self, targets, thresh = 0.5):
        res = []
        for b in range(len(targets)):
            
            temp_dict = dict()
            
            scores = targets[b]['scores'].detach().cpu().numpy()
            probs = targets[b]['probs'].detach().cpu().numpy()
            labels = targets[b]['labels'].detach().cpu().numpy()
            coeffs = targets[b]['boxes'].detach().cpu().numpy()
            assoc = targets[b]['assoc'].detach().cpu().numpy()
            
            selecteds = (scores>thresh) * (labels!=0)###把大于阈值的筛选出来
            # selecteds = probs[:,1] > thresh
            
            detected_scores = scores[selecteds]
            detected_labels = labels[selecteds]
            detected_coeffs = coeffs[selecteds,...]
            
            coef_all_roads = np.zeros((196,200,3),np.float32)
            if len(detected_scores) > 0:
                
                temp_dict['scores'] = detected_scores
                temp_dict['boxes'] = detected_coeffs
                temp_dict['assoc'] = assoc
                temp_dict['labels'] = detected_labels
    #          
                to_merge = {'assoc': assoc,'boxes':detected_coeffs}
                merged = get_merged_coeffs(to_merge)#在detected_coeffs的基础上 ，对于要连接的两条line，出发line的末端点和到达line的起始点都改成了他们的中点（点的总数没变）
                temp_dict['merged_coeffs'] = merged
                
                res_coef_list=[]
                res_interpolated_list=[]
                for k in range(len(detected_scores)):

                    control = detected_coeffs[k]

                    interpolated = bezier.interpolate_bezier(control,100)
                    
                    res_interpolated_list.append(np.copy(interpolated))
                    
                    coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)#[2,2]
                    line2 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))
                    res_coef_list.append(line2)
                    coef_all_roads = coef_all_roads + np.float32(line2)
                
                
                temp_dict['interpolated_points'] = res_interpolated_list #长度为分数大于thresh的预测数量（N），每个元素shape为(100,2)
                
                temp_dict['coef_lines'] = res_coef_list#长度为N。每个元素shape为(196, 200, 3)，值是0和1
                temp_dict['coef_all_roads'] = coef_all_roads#把上一行的N个图片全加起来
                
                merged_interpolated_list=[]
                for k in range(len(merged)):
                    control = merged[k]
                    coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
                    
                    merged_interpolated = bezier.interpolate_bezier(control,100)
                    
                    merged_interpolated_list.append(np.copy(merged_interpolated))
                

                temp_dict['merged_interpolated_points'] = merged_interpolated_list#与temp_dict['interpolated_points']类似,区别是插值用的控制点根据连接关系做过调整
                
                
                
            else:
                
                # logging.error('DETECTED NOTHING. That means no prediction score is higher than thresh. It usually happends at ther beginning of training.')
                temp_dict['scores'] = []
                temp_dict['interpolated_points'] = []
                temp_dict['scores'] = []
                temp_dict['boxes'] = []
                temp_dict['lines'] = []
                temp_dict['coef_lines'] = []
                temp_dict['all_roads'] = []
                temp_dict['coef_all_roads'] = []
                temp_dict['labels'] = []
                temp_dict['assoc'] = []
    
                temp_dict['merged_interpolated_points'] = []
                temp_dict['merged_coeffs'] = []
            
            res.append(temp_dict)
            
        return res

    def get_final_layer(self, outs):
        for key in ['pred_logits', 'pred_spline', 'assoc_features']:
            outs[key] = outs[key][-1]
        return outs

    def hausdorff_match(self,out,target):
        est_coefs = out['boxes']
        
        orig_coefs = target['control_points'].cpu().numpy()
        orig_coefs = np.reshape(orig_coefs, (-1, int(orig_coefs.shape[-1]/2),2))
        
        interpolated_origs = []
        
        for k in range(len(orig_coefs)):
            inter = bezier.interpolate_bezier(orig_coefs[k],100)
            interpolated_origs.append(np.copy(inter))
            
        if len(est_coefs) == 0:
            return None,None, interpolated_origs
        
        
        dist_mat = np.mean(np.sum(np.square(np.expand_dims(est_coefs,axis=1) - np.expand_dims(orig_coefs,axis=0)),axis=-1),axis=-1)#(预测数量，GT数量)
        
        ind = np.argmin(dist_mat, axis=-1)#(预测数量,)
        min_vals = np.min(dist_mat,axis=-1)
        
            
        return min_vals, ind, interpolated_origs 

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        # import pdb;pdb.set_trace()
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # import pdb;pdb.set_trace()
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # import pdb;pdb.set_trace()
        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan,
                                                               :4], bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # pts_preds = pts_preds
        # import pdb;pdb.set_trace()
        
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2),pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        # import pdb;pdb.set_trace()
        loss_pts = self.loss_pts(
            pts_preds[isnotnan,:,:], normalized_pts_targets[isnotnan,
                                                            :,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)
        dir_weights = pts_weights[:, :-self.dir_interval,0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.dir_interval:,:] - denormed_pts_preds[:,:-self.dir_interval,:]
        pts_targets_dir = pts_targets[:, self.dir_interval:,:] - pts_targets[:,:-self.dir_interval,:]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan,:,:], pts_targets_dir[isnotnan,
                                                                          :,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    # def loss_straight(self, outputs, targets, indices, log=True):
    #     idx = self._get_src_permutation_idx(indices)
    #     src_boxes = outputs['pred_spline'][idx].contiguous().view(-1,self.num_control_points,2)#(20,50,2)
        
    #     target_classes_o = torch.cat([t["labels"].squeeze(0)[J] for t, (_, J) in zip(targets, indices)])
    #     mask = (target_classes_o==1)
        
    #     src_boxes = src_boxes[mask]#(11,50,2)
            
    #     inter_points = torch.matmul(self.bezierB.expand(src_boxes.size(0),-1,-1),src_boxes)
        
    #     X = inter_points[:,:,0].unsqueeze(2)
    #     X = torch.cat([X, torch.ones_like(X)], dim=2)
    #     y = inter_points[:,:,1].unsqueeze(2)

    #     distance_sum = 0
    #     for i in range(X.shape[0]):
    #         solution = torch.linalg.lstsq(X[i], y[i]).solution
    #         m, b = solution[:, 0]
            
    #         distances = torch.abs(m * inter_points[i, :, 0] - inter_points[i, :, 1] + b) / torch.sqrt(m ** 2 + 1)
    #         distance_sum += torch.sum(distances)
        
    #     losses = {}
    #     losses['loss_straight'] = distance_sum
    #     return losses
    
    def loss_straight(self, outputs, targets, indices, log=True):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_spline'][idx].contiguous().view(-1,self.num_control_points,2)#(20,5,2)
        target_classes_o = torch.cat([t["labels"].squeeze(0)[J] for t, (_, J) in zip(targets, indices)])
        mask = (target_classes_o==1)
        src_boxes = src_boxes[mask]#(11,5,2)
        src_points = torch.matmul(self.bezierC.expand(src_boxes.size(0),-1,-1),src_boxes)#(11,5,2)
        pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)
        pi = pi.to(src_points.device)

        loss = 0
        #原本是计算一阶导数的变化量作为loss；更改为计算角度(弧度）的变化量
        for i in range(src_points.shape[0]):
            points = src_points[i]
            dx = points[1:, 0] - points[0, 0]
            dy = points[1:, 1] - points[0, 1]
            slopes = dy / dx
            slope_degrees = torch.atan(slopes)
            dslopes = slope_degrees[1:] - slope_degrees[:-1]

            for j in range(len(dslopes)):
                if dslopes[j] < -pi/2:
                    dslopes[j] = dslopes[j] + pi
                elif dslopes[j] > pi/2:
                    dslopes[j] = dslopes[j] - pi

            dslopes_h = torch.max(dslopes - 0.5,torch.zeros_like(dslopes).cuda())
            dslopes_l = torch.min(dslopes + 0.5,torch.zeros_like(dslopes).cuda())
            loss += torch.sum(dslopes_h**2 + dslopes_l**2)
            
        losses = {}
        if src_boxes.shape[0]!=0:
            losses['loss_straight'] = loss/src_boxes.shape[0]
        else:
            losses['loss_straight'] = torch.tensor([0.0]).to(device=src_boxes.device)
        return losses       
        
    def loss_connect(self, outputs, targets, indices, log=True):
        idx = self._get_src_permutation_idx(indices)
        _, target_ids = self._get_tgt_permutation_idx(indices)
        est = outputs['pred_spline'][idx].contiguous().view(-1,self.num_control_points,2)
        lab = targets[0]['con_matrix'].squeeze(0)
        lab = lab.float()
        lab = lab[target_ids,:]
        lab = lab[:,target_ids]
        loss_connect = 0
        for i in range(lab.shape[0]):
            for j in range(lab.shape[1]):
                if lab[i][j]==1:
                    loss_connect += F.l1_loss(est[i][-1],est[j][0],reduction='sum')     
        
        losses = {}
        if lab.sum()!=0:
            losses['loss_connect'] = loss_connect/lab.sum()
        else:
            losses['loss_connect'] = torch.tensor([0.0]).to(device=est.device)
        return losses
    
    #原来的版本
    # def loss_smooth(self, outputs, targets, indices, log=True):
    #     target_classes = targets[0]['labels'].squeeze(0)
    #     mask_2 = (target_classes==2)
        
    #     idx = self._get_src_permutation_idx(indices)
    #     src_boxes = outputs['pred_spline'][idx].contiguous().view(-1,self.num_control_points,2)
    #     src_boxes = src_boxes[mask_2]
        
    #     src_points = torch.matmul(self.bezierB.expand(src_boxes.size(0),-1,-1),src_boxes)

    #     loss_curvature = 0
    #     for i in range(src_points.shape[0]):
    #         spline = src_points[i]
    #         loss_curvature += curvature_loss(spline)

    #     losses = {}
    #     if src_points.shape[0]!=0:
    #         losses['loss_smooth'] = loss_curvature/(src_points.shape[0])
    #     else:
    #         losses['loss_smooth'] = torch.tensor([0.0]).to(device=src_points.device)
    #     return losses 

    #还要设置bezier采样点
    def loss_smooth(self, outputs, targets, indices, log=True):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_spline'][idx].contiguous().view(-1,self.num_control_points,2)#(20,5,2)
        target_classes_o = torch.cat([t["labels"].squeeze(0)[J] for t, (_, J) in zip(targets, indices)])
        mask = (target_classes_o==2)
        src_boxes = src_boxes[mask]#(11,5,2)
        
        loss = 0
        for i in range(src_boxes.shape[0]):
            points = src_boxes[i]
            dx = points[1:, 0] - points[:-1, 0]
            dy = points[1:, 1] - points[:-1, 1]
            slopes = dy / dx
            dslopes = slopes[1:] - slopes[:-1]
            dslopes_dx = dslopes / dx[:-1]
            dslopes_dx_degrees = torch.atan(dslopes_dx)
            ddslopes = dslopes_dx_degrees[1:] - dslopes_dx_degrees[:-1]

            loss += torch.sum(ddslopes ** 2)
        losses = {}
        if src_boxes.shape[0]!=0:
            losses['loss_smooth'] = loss/src_boxes.shape[0]
        else:
            losses['loss_smooth'] = torch.tensor([0.0]).to(device=src_boxes.device)
        return losses   
        
    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"].squeeze(0)[J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.num_stsu_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        # empty_weight = torch.tensor([1.0,1.0,0.3],device=src_logits.device)
        empty_weight = torch.tensor([0.3,1.0,1.0],device=src_logits.device)
        #0代表没有被选中与GT训练 1代表lane_center, 2代表lane_connector
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_assoc(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        _, idx = self._get_src_permutation_idx(indices)
        _, target_ids = self._get_tgt_permutation_idx(indices)
        
        lab = targets[0]['con_matrix'].squeeze(0)
        lab = lab.float()
        lab = lab[target_ids,:]
        lab = lab[:,target_ids]
        
        est = outputs['pred_assoc']
        
        mask = lab*3 + (1-lab)
        
        loss_ce = torch.mean(F.binary_cross_entropy_with_logits(est.view(-1),lab.view(-1),weight=mask.float().view(-1)))
        losses = {'loss_assoc': loss_ce}
        

        src_boxes = outputs['pred_spline'][0][idx]
        src_boxes = src_boxes.view(-1, int(src_boxes.shape[-1]/2), 2)
        
        start_points = src_boxes[:,0,:].contiguous()
        end_points = src_boxes[:,-1,:].contiguous()
        
        my_dist = torch.cdist(end_points, start_points, p=1)
        
        #GT认为应当连接的两条line，第一条line的末端和第二条line的起点距离过大会有惩罚；GT认为不应当连接的两条line，第一条line的末端和第二条line的起点距离如果过小会有惩罚
        cost_end = 2*my_dist*lab - 3*torch.min(my_dist - 0.05,torch.zeros_like(my_dist).cuda())*(1-lab)
#        losses = {'loss_end_match': cost_end.sum()/(lab.sum() + 0.0001)}
        losses['loss_end_match']= cost_end.mean()

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_polyline(self, outputs, targets, indices):
        """
        由n个控制点插值得到的50个点的曲线的坐标误差 
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_spline'][idx].contiguous().view(-1,self.num_control_points,2)
        target_boxes = torch.cat([t['control_points'].squeeze(0)[i] for t, (_, i) in zip(targets, indices)], dim=0)
       
        target_boxes = target_boxes.contiguous().view(-1,self.num_control_points,2)
        
        inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
        
        target_points = torch.matmul(self.bezierA.expand(target_boxes.size(0),-1,-1),target_boxes)
        
        cost_bbox = torch.cdist(inter_points, target_points, p=1)
        
        min0 = torch.mean(torch.min(cost_bbox,dim=1)[0],dim=-1)
        min1 = torch.mean(torch.min(cost_bbox,dim=2)[0],dim=-1)
        
        losses = {}
        losses['loss_polyline'] = torch.mean(min0 + min1)
        return losses
    

    def loss_control_points(self, outputs, targets, indices):
        """
        控制点的坐标误差
        """
        assert 'pred_spline' in outputs
    
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_spline'][idx]
        target_boxes = torch.cat([t['control_points'].squeeze(0)[i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.mean()
        return losses
        
    
    # def loss_endpoints(self, outputs, targets, indices):
    #     """
    #     """
    #     assert 'pred_endpoints' in outputs
    
    #     idx = self._get_src_permutation_idx(indices)
    #     src_boxes = outputs['pred_endpoints'][idx]
    #     target_boxes = torch.cat([t['endpoints'].squeeze(0)[i] for t, (_, i) in zip(targets, indices)], dim=0)

    #     loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

    #     losses = {}
    #     losses['loss_endpoints'] = loss_bbox.mean()

    #     return losses


    def focal_loss(self, outputs, targets, indices, log=True):
        
        alpha = 0.8
        gamma = 2
        epsilon = 0.00001
        beta = 4
        
        idx = self._get_src_permutation_idx(indices)
        
        src_boxes = outputs['pred_boxes'][idx].view(-1,self.num_control_points,2)
        
        inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
        
        estimated_traj = bezier.gaussian_line_from_traj(inter_points)
        
        target_boxes = torch.cat([t['smoothed'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        target_boxes = torch.clamp(target_boxes,0,1)
        
        hard_labels = (target_boxes > 0.7).float()
        
        labels=target_boxes
        y_pred = estimated_traj
        
        
        L=-hard_labels*alpha*torch.pow((1-y_pred),gamma)*torch.log(y_pred + epsilon)-\
          (1-hard_labels)*(1-alpha)*torch.pow(1-labels,beta)*torch.pow(y_pred,gamma)*torch.log(1-y_pred + epsilon)
   
        losses = {}
        losses['focal'] = L.mean()

        return losses

    def get_loss(self, loss, outputs, targets, indices,  **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'assoc': self.loss_assoc,
        
            'boxes': self.loss_control_points,
            'loss_polyline': self.loss_polyline,
            'focal': self.focal_loss,
            
            'straight':self.loss_straight,
            'connect':self.loss_connect,
            'smooth':self.loss_smooth
            # 'endpoints': self.loss_endpoints
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices,  **kwargs)



    @force_fp32(apply_to=('preds_dicts'))
    def loss_stsu(self,
             target,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        losses = {}
        outputs = {}
        outputs['pred_logits'] = preds_dicts['pred_logits'] # [4, bs, 100, 3]
        outputs['pred_spline'] = preds_dicts['pred_spline'] # [4, bs, 100, 5*2]
        outputs['assoc_features'] = preds_dicts['assoc_features'] # [4, bs, 100, 128]

        #计算每一层layer的loss
        for i in range(outputs['pred_logits'].shape[0]):
            layer_loss = {}
            output = {}
            output['pred_logits'] = outputs['pred_logits'][i]
            output['pred_spline'] = outputs['pred_spline'][i]
            output['assoc_features'] = outputs['assoc_features'][i]
            
            indices_static, _ = self.matcher(output, target)
            output = self.get_assoc_estimates(output,indices_static)
             #outputs['pred_assoc'].shape = torch.Size([1, 16, 16])  #bs=1

            for loss in self.stsu_losses:
                layer_loss.update(self.get_loss(loss, output, target, indices_static))
            
            for key, value in layer_loss.items():
                if key in losses:
                    losses[key] += value
                else:
                    losses[key] = value
        
        return losses

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        # import pdb;pdb.set_trace()
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_pts_preds  = preds_dicts['all_pts_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        enc_pts_preds  = preds_dicts['enc_pts_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
        #     dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        # import pdb;pdb.set_trace()
        # gt_bboxes_list = [
        #     gt_bboxes.to(device) for gt_bboxes in gt_bboxes_list]
        gt_bboxes_list = [
            gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_pts_list = [
            gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        if self.gt_shift_pts_pattern == 'v0':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v1':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v2':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v3':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v4':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in gt_vecs_list]
        else:
            raise NotImplementedError
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_pts_list = [gt_pts_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        # import pdb;pdb.set_trace()
        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,all_pts_preds,
            all_gt_bboxes_list, all_gt_labels_list,all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            # TODO bug here
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, enc_losses_pts, enc_losses_dir = \
                self.loss_single(enc_cls_scores, enc_bbox_preds, enc_pts_preds,
                                 gt_bboxes_list, binary_labels_list, gt_pts_list,gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_losses_iou'] = enc_losses_iou
            loss_dict['enc_losses_pts'] = enc_losses_pts
            loss_dict['enc_losses_dir'] = enc_losses_dir

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1],
                                           losses_iou[:-1],
                                           losses_pts[:-1],
                                           losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            num_dec_layer = num_dec_layer + 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            pts = preds['pts']

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list

