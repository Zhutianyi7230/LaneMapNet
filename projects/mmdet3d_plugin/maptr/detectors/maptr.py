import copy
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmcv.runner import force_fp32, auto_fp16
import torch
import numpy as np
from .utils import BinaryConfusionMatrix

@DETECTORS.register_module()
class MapTR(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 use_prev_ref_pts = False
                 ):

        super(MapTR,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.use_prev_ref_pts = use_prev_ref_pts
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
            'prev_ref_pts': None
        }
        # self.confusion = BinaryConfusionMatrix(1)####


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          target=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        #gt_bboxes_3d[0].fixed_num_sampled_points.shape = torch.Size([10, 20, 2])
        #gt_labels_3d[0].shape = torch.Size([10])

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        del outs['last_reference']#这一项只有预测的时候用到。

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        
        ###
        loss_inputs_stsu = [target,outs]
        losses_stsu = self.pts_bbox_head.loss_stsu(*loss_inputs_stsu, img_metas=img_metas)

        ###调整losses_stsu中各项的权重
        losses_stsu['loss_ce'] = losses_stsu['loss_ce'] * 1
        losses_stsu['loss_bbox'] = losses_stsu['loss_bbox'] * 30
        losses_stsu['loss_polyline'] = losses_stsu['loss_polyline'] * 30
        losses_stsu['loss_assoc'] = losses_stsu['loss_assoc'] * 20
        losses_stsu['loss_end_match'] = losses_stsu['loss_end_match'] * 20
        if 'loss_straight' in losses_stsu.keys():
            losses_stsu['loss_straight'] = losses_stsu['loss_straight']*10
        if 'loss_connect' in losses_stsu.keys():
            losses_stsu['loss_connect'] = losses_stsu['loss_connect'] 
        
        losses.update(losses_stsu)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    # def obtain_history_bev(self, imgs_queue, img_metas_list):
    #     """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
    #     """
    #     # self.eval()

    #     with torch.no_grad():
    #         # 禁用计算梯度
    #         for param in self.pts_bbox_head.parameters():
    #             param.requires_grad = False
    #         for param in self.img_backbone.parameters():
    #             param.requires_grad = False    
    #         for param in self.img_neck.parameters():
    #             param.requires_grad = False  
                
    #         prev_bev = None
    #         bs, len_queue, num_cams, C, H, W = imgs_queue.shape
    #         imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
    #         img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            
    #         for i in range(len_queue):
    #             img_metas = [each[i] for each in img_metas_list]
    #             # img_feats = self.extract_feat(img=img, img_metas=img_metas)
    #             img_feats = [each_scale[:, i] for each_scale in img_feats_list]
    #             prev_bev = self.pts_bbox_head.forward(
    #                 img_feats, img_metas, prev_bev, only_bev=True)
    #         # self.train()     
    #         # 恢复计算梯度
    #         for param in self.pts_bbox_head.parameters():
    #             param.requires_grad = True
    #         for param in self.img_backbone.parameters():
    #             param.requires_grad = True    
    #         for param in self.img_neck.parameters():
    #             param.requires_grad = True             
    #     return prev_bev

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      target=None,####为了STSU添加的
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        #假设len_queue = 2
        #img.shape = torch.Size([1, len_que, 6, 3, 480, 800])
        #target.keys()=dict_keys(['center_img', 'orig_center_img', 'labels', 'roads', 'control_points', 'con_matrix', 'endpoints', 'origs', 'outgoings', 'incomings', 'left_traffic'])
        # print('--------------------begin a iter-----------------------------')
        # import pdb
        # pdb.set_trace()
        B = img.size(0)
        len_queue = img.size(1)
        N= img.size(2)
        C = img.size(3)
        H =img.size(4)
        W = img.size(5)
        prev_img = img[:, :-1, ...] # torch.Size([1, 1, 6, 3, 480, 800])
        curr_img = img[:, -1, ...].unsqueeze(1) # torch.Size([1, 1, 6, 3, 480, 800])
        img = img.reshape([B*len_queue,N,C,H,W])

        prev_bev = None
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue>1 else None

        curr_img_metas = [each[len_queue-1] for each in img_metas]
        prev_img_metas = [each[0] for each in img_metas]
        
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        #img_feats[0].shape = torch.Size([2, 6, 256, 15, 25])

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev, [target])

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, target=None, **kwargs):
        #原版img[0].shape = torch.Size([1, 6, 3, 480, 800])
        #原版img_metas[0][0].keys()

        #img.shape = torch.Size([1,len_queue, 6, 3, 480, 800])
        #img_metas[0][1].keys(),前一个0代表bs,后一个1代表时序
        
        # ###
        # print('begin a test')
        # import pdb
        # pdb.set_trace()
        # ###
        
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            # self.prev_frame_info['prev_bev'] = None
            self.prev_frame_info['prev_ref_pts'] = None
        # # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # # do not use temporal information
        # if not self.video_test_mode:
        #     self.prev_frame_info['prev_bev'] = None
        if not self.use_prev_ref_pts:
            self.prev_frame_info['prev_ref_pts'] = None

        # # Get the delta of ego position and angle between two timestamps.
        # tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        # tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # if self.prev_frame_info['prev_bev'] is not None:
        #     img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
        #     img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        # else:
        #     img_metas[0][0]['can_bus'][-1] = 0
        #     img_metas[0][0]['can_bus'][:3] = 0

        # new_prev_bev, bbox_results = self.simple_test(
        #     img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)     
        B = img.size(0)
        len_queue = img.size(1)
        N= img.size(2)
        C = img.size(3)
        H =img.size(4)
        W = img.size(5)
        img = img.reshape([B*len_queue,N,C,H,W])
        curr_img_metas = [each[len_queue-1] for each in img_metas]
        prev_img_metas = [each[0] for each in img_metas]

        bbox_results = self.simple_test(
            img_metas, img, prev_bev=None, prev_ref_pts=self.prev_frame_info['prev_ref_pts'], target=target, **kwargs) #last_reference.shape = torch.Size([1, 1000, 2])

        # During inference, we save the BEV features and ego motion of each timestamp.
        # self.prev_frame_info['prev_pos'] = tmp_pos
        # self.prev_frame_info['prev_angle'] = tmp_angle
        # self.prev_frame_info['prev_bev'] = new_prev_bev

        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()

        return result_dict
    
    def pred2result_stsu(self, bboxes, labels, assoc, scores, interpolated_points, merged_interpolated_points):
        """Convert detection results to a list of numpy arrays.
        """
        if len(bboxes)!=0:
            result_dict = dict(
                boxes=torch.from_numpy(bboxes).cpu(),
                labels=torch.from_numpy(labels).cpu(),
                assoc=torch.from_numpy(assoc).cpu(),
                scores = torch.from_numpy(scores).cpu(),
                interpolated_points=torch.from_numpy(np.array(interpolated_points)).cpu(),
                merged_interpolated_points = torch.from_numpy(np.array(merged_interpolated_points)).cpu()
                )
        else:
            result_dict = dict(
                boxes=[],
                labels=[],
                assoc=[],
                scores=[],
                interpolated_points=[],
                merged_interpolated_points=[]
            )

        return result_dict
    
    def simple_test_pts(self, x, img_metas, prev_bev=None, prev_ref_pts=None, rescale=False, target=None):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, prev_ref_pts=prev_ref_pts)
        del outs['last_reference']

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        static_thresh = 0.5#初步阈值。选0.5是因为score基本都大于0.5，所以0.5以下区分的意义不大
        
        outs = self.pts_bbox_head.get_final_layer(outs)#test时只取最后一层decoder的结果
        outs = self.pts_bbox_head.thresh_and_assoc_estimates(outs, static_thresh)#对STSU输出结果进行thresh和assoc
        
        base_postprocessed = self.pts_bbox_head.postprocess(outs)
        out = self.pts_bbox_head.get_selected_estimates(base_postprocessed,static_thresh)
        
        bbox_results = [
            self.pred2result(bboxes, scores, labels, pts)
            for bboxes, scores, labels, pts in bbox_list
        ]#原maptr的预测结果

        stsu_results = [self.pred2result_stsu(out_['boxes'], out_['labels'], out_['assoc'], out_['scores'], out_['interpolated_points'], out_['merged_interpolated_points'])
                        for out_ in out]#
        
        return bbox_results,stsu_results
    
    def simple_test(self, img_metas, img=None, prev_bev=None, prev_ref_pts=None, rescale=False, target=None, **kwargs):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]

        bbox_pts, stsu_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, prev_ref_pts, rescale=rescale, target=target)
        for result_dict, pts_bbox, pts_stsu in zip(bbox_list, bbox_pts, stsu_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['pts_stsu'] = pts_stsu
  
        return bbox_list


@DETECTORS.register_module()
class MapTR_fp16(MapTR):
    """
    The default version BEVFormer currently can not support FP16. 
    We provide this version to resolve this issue.
    """
    # @auto_fp16(apply_to=('img', 'prev_bev', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      prev_bev=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # import pdb;pdb.set_trace()
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev=prev_bev)
        losses.update(losses_pts)
        return losses


    def val_step(self, data, optimizer):
        """
        In BEVFormer_fp16, we use this `val_step` function to inference the `prev_pev`.
        This is not the standard function of `val_step`.
        """

        img = data['img']
        img_metas = data['img_metas']
        img_feats = self.extract_feat(img=img,  img_metas=img_metas)
        prev_bev = data.get('prev_bev', None)
        prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        return prev_bev
