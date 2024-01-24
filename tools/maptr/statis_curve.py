import argparse
import mmcv
import os
import torch
import glob
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2])) 

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import scipy.ndimage as ndimage
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
from projects.mmdet3d_plugin.bevformer.apis.test import hausdorff_match
import  projects.mmdet3d_plugin.datasets.bezier  as bezier

device='cuda:0'

def compute_curvature(points, file_path):
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
    curvature = curvature.cpu().numpy()

    with open(file_path,'a') as f:
        for d in curvature:
            f.write(str(d) + ' ')
        f.write('\n')

def derivatives(points, file_path):
    pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)
    dx = points[1:, 0] - points[0, 0]
    dy = points[1:, 1] - points[0, 1]
    # dx = points[1:, 0] - points[:-1, 0]
    # dy = points[1:, 1] - points[:-1, 1]
    slopes = dy / dx
    slope_degrees = torch.atan(slopes)   
    dslopes = slope_degrees[1:] - slope_degrees[:-1]
    #夹角实在【-90，90】之间的
    modified_values = []
    for val in dslopes:
        if val < -pi/2:
            val += pi
        elif val > pi/2:
            val -= pi
        modified_values.append(val)
    dslopes = torch.stack(modified_values) 
    dslopes = dslopes.cpu().numpy()
    
    with open(file_path,'a') as f:
        for d in dslopes:
            f.write(str(d) + ' ')
        f.write('\n')
    

def second_derivatives(points, file_path):
    pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)
    # Calculate the differences in x and y coordinates
    dx = points[1:, 0] - points[:-1, 0]
    dy = points[1:, 1] - points[:-1, 1]

    # Avoid division by zero for vertical lines
    dx = torch.clamp(dx, min=1e-8)

    # Calculate slopes
    slopes = dy / dx

    # Calculate the change in slope between consecutive segments
    dslopes = slopes[1:] - slopes[:-1]
    second_derivatives = dslopes / dx[:-1]
    
    second_derivatives_degree = torch.atan(second_derivatives)

    modified_values = []
    for val in second_derivatives_degree:
        if val > 3.0:
            val -= pi
        elif val < -3.0:
            val += pi
        modified_values.append(val)
    second_derivatives_degree = torch.stack(modified_values) 
      
    second_derivatives_degree = second_derivatives_degree/pi*180.0     
    delta_second_derivatives_degree = second_derivatives_degree[1:] - second_derivatives_degree[:-1]
    

    import pdb;pdb.set_trace()
    
    delta_second_derivatives_degree = delta_second_derivatives_degree.cpu().numpy()
    points = points.cpu().numpy()

    with open(file_path,'a') as f:
        for d in delta_second_derivatives_degree:
            f.write(str(d) + ' ')
        f.write('\n')


def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=2000, help='samples to visualize')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=["fixed_num_pts"],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join('./work_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                'vis_pred_stsu')
    # create vis_label dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_pred dir: {args.show_dir}')


    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    cfg.model.pts_bbox_head.bbox_coder.max_num=15 # TODO this is a hack
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    logger.info('BEGIN vis test dataset samples gt label & pred')

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    bezierB = bezier.bezier_matrix(n_control=5,n_int=20)
    bezierB = bezierB.cuda()
    bezierC = bezier.bezier_matrix(n_control=5,n_int=5)
    bezierC = bezierC.cuda()

    for i, data in enumerate(data_loader):
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            prog_bar.update()  
            continue
        
        # with torch.no_grad():
        #     result = model(return_loss=False, rescale=True, **data)
        
        ###gt
        #target.keys() = dict_keys(['center_img', 'labels', 'roads', 'control_points', 'con_matrix'])
        target = data['target']
        
        target_classes = target['labels'].squeeze(0)
        mask_2 = (target_classes==2)
        target_boxes = target['control_points'].squeeze(0)[mask_2].view(-1,5,2).to(device)
        target_points = torch.matmul(bezierB.expand(target_boxes.size(0),-1,-1),target_boxes)
        
        mask_1 = (target_classes==1)
        target_boxes_1 = target['control_points'].squeeze(0)[mask_1].view(-1,5,2).to(device)
        target_points_1 = torch.matmul(bezierC.expand(target_boxes_1.size(0),-1,-1),target_boxes_1)
        
        for j in range(target_points_1.shape[0]):    
            spline_1 = target_points_1[j]
            derivatives(spline_1,'tools/maptr/statics/delta_derivatives_degree_target2.txt')
        
        # for j in range(target_points.shape[0]):
        #     spline = target_points[j]#(20,2)
        #     second_derivatives(spline, 'tools/maptr/statics/delta_second_derivatives_degree_target.txt')  
            # compute_curvature(spline, 'tools/maptr/statics/curvature_target.txt')         
        

        # ###output
        # result = result[0]['pts_stsu']
        # output_classes = result['labels']
        # mask_2 = (output_classes==2)
        # output_boxes = result['boxes'].squeeze(0)[mask_2].view(-1,5,2).to(device)
        # output_points = torch.matmul(bezierB.expand(output_boxes.size(0),-1,-1),output_boxes)
        
        # for j in range(output_points.shape[0]):
        #     spline = output_points[j]#(20,2)
        #     second_derivatives(spline, 'tools/maptr/statics/delta_second_derivatives_degree_output.txt')  
        #     # compute_curvature(spline, 'tools/maptr/statics/curvature_output.txt') 
        
        


        prog_bar.update()

if __name__ == '__main__':
    main()
