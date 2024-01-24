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

def distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def closest_point_to_a(a, b, c):
    distance_b = distance(a, b)
    distance_c = distance(a, c)
    if distance_b <= distance_c:
        return b
    elif distance_c < distance_b:
        return c
    
def my_color_line_maker(points,endpoints,size=(196,200)):
    if len(endpoints) == 4:
        endpoints = np.reshape(endpoints,[2,2])

    res = np.zeros((size[0],size[1],3))
    #把points中100个点所在位置的像素设为1
    for k in range(len(points)):
        res[np.min([int(points[k][1]*size[0]),int(size[0]-1)]),np.min([int(points[k][0]*size[1]),int(size[1]-1)])] = 1
    
    base_start = np.zeros((res.shape[0],res.shape[1]))
    base_start[np.min([int(endpoints[0,1]*size[0]),int(size[0]-1)]),np.min([int(endpoints[0,0]*size[1]),int(size[1]-1)])] = 1#把线条的起始位置点设为1

    struct = ndimage.generate_binary_structure(2, 2)
    
    dilated = ndimage.binary_dilation(base_start>0, structure=struct)#对base_start上唯一的True的像素点膨胀，最终周围9个点都为True
    
    res[dilated,0] = 0
    res[dilated,1] = 1
    res[dilated,2] = 0
    
    base_end = np.zeros((res.shape[0],res.shape[1]))
    base_end[np.min([int(endpoints[1,1]*size[0]),int(size[0]-1)]),np.min([int(endpoints[1,0]*size[1]),int(size[1]-1)])] = 1#把线条的末端位置点设为1
    
    # struct = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(base_end>0, structure=struct)#对base_end上唯一的True的像素点膨胀，最终周围9个点都为True
    
    res[dilated,0] = 1
    res[dilated,1] = 0
    res[dilated,2] = 0
    
    
    return np.uint8(255*res)

def get_vertices(adj):

    ins = []
    outs = []
    
    for k in range(len(adj)):
        for m in range(len(adj)):
            if adj[k,m] > 0.5:
                if len(ins) > 0:
                    ins_exists = False
                    out_exists = False
    
                    for temin in range(len(ins)):
                        if k in ins[temin]:
                            if not (m in outs[temin]):
                                outs[temin].append(m)
                            ins_exists=True
                            break
                    if not ins_exists:
                        for temin in range(len(outs)):
                            if m in outs[temin]:
                                if not (k in ins[temin]):
                                    ins[temin].append(k)
                                out_exists=True
                                break  
                        if not out_exists:
                            ins.append([k])
                            outs.append([m])
                else:
                    ins.append([k])
                    outs.append([m])

    return ins, outs      

def get_merged_network(targets):
    
    coeffs = targets['boxes']
    assoc = targets['assoc']
    diag_mask = np.eye(len(assoc))
    diag_mask = 1 - diag_mask
    assoc = assoc*diag_mask
    corrected_coeffs = np.copy(coeffs)
    
    ins, outs = get_vertices(assoc)
    
    for k in range(len(ins)):
        all_points=[]
        for m in ins[k]:
            all_points.append(corrected_coeffs[m,-1])
            
        for m in outs[k]:
            all_points.append(corrected_coeffs[m,0])
            
        
        av_p = np.mean(np.stack(all_points,axis=0),axis=0)
        
        for m in ins[k]:
            corrected_coeffs[m,-1] = av_p
            
        for m in outs[k]:
            corrected_coeffs[m,0] = av_p
            
    lines=[]
    for k in range(len(corrected_coeffs)):        
        control = corrected_coeffs[k]
        coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
        interpolated = bezier.interpolate_bezier(control)
        line = np.float32(my_color_line_maker(interpolated,coef_endpoints,size=(400,200)))/255
        lines.append(line)    
    
    return lines

def get_assoc_indexes_and_curves(ins, outs, coeffs):
    assoc_indexes = []
    assoc_curves = []
    for k in range(len(ins)):
        for m in ins[k]:
            for n in outs[k]:
                assoc_indexes.append([m,n])
                assoc_curves.append(np.concatenate([coeffs[m],coeffs[n]],axis=0))
    return assoc_indexes, assoc_curves

#如果直接把相连接的curve对应的起点和终点取平均值merge的话，即使是target的效果也很差。
#所以考虑把所有的控制点都保留，两条相连的线就有2n个控制点
def get_target_merged_network(targets):
    
    coeffs = targets['control_points'].squeeze(0)
    coeffs = coeffs.reshape([coeffs.shape[0],-1,2])
    assoc = targets['con_matrix'].squeeze(0)
    diag_mask = np.eye(len(assoc))
    diag_mask = 1 - diag_mask
    assoc = assoc*diag_mask
    corrected_coeffs = np.copy(coeffs)
    
    ins, outs = get_vertices(assoc)
    assoc_indexes, assoc_curves = get_assoc_indexes_and_curves(ins, outs, coeffs)
            
    lines=[]
    for k in range(len(assoc_curves)):        
        control = assoc_curves[k]
        coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
        interpolated = bezier.interpolate_bezier(control)
        line = np.float32(my_color_line_maker(interpolated,coef_endpoints,size=(400,200)))/255
        lines.append(line)    

    return lines

def get_endpoints_from_coeffs(coeffs):
    start = coeffs[:,:2]
    end = coeffs[:,-2:]
    return np.concatenate([start,end],axis=-1)

def add_endpoints_to_line(ar,endpoints):
    if len(endpoints) == 4:
        endpoints = np.reshape(endpoints,[2,2])
    size = ar.shape
    res = np.zeros((ar.shape[0],ar.shape[1],3))
    res[ar > 0] = 1
    
    # logging.error('AR SHAPE ' + str(ar.shape))
    # logging.error('ENDPOINTS SHAPE ' + str(endpoints.shape))
    # logging.error('ENDPOINTS ' + str(endpoints))
    
    base_start = np.zeros((ar.shape[0],ar.shape[1]))
    base_start[np.min([int(endpoints[0,1]*size[0]),int(size[0]-1)]),np.min([int(endpoints[0,0]*size[1]),int(size[1]-1)])] = 1
    
    # struct = ndimage.generate_binary_structure(5, 2)
    struct = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(base_start>0, structure=struct)
    
    res[dilated,0] = 0
    res[dilated,1] = 1
    res[dilated,2] = 0
    
    base_end = np.zeros((ar.shape[0],ar.shape[1]))
    base_end[np.min([int(endpoints[1,1]*size[0]),int(size[0]-1)]),np.min([int(endpoints[1,0]*size[1]),int(size[1]-1)])] = 1
    
    # struct = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(base_end>0, structure=struct)
    
    res[dilated,0] = 1
    res[dilated,1] = 0
    res[dilated,2] = 0
    
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),0] = 1
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),1] = 0
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),2] = 0
    
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),0] = 0
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),1] = 0
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),2] = 1
    
    return res

def vis_est(result, save_path, name=None):
    
    result = result['pts_stsu']
    import pdb;pdb.set_trace()
    #重新merge画连接起来的
    merged = get_merged_network(result)
    if len(merged) > 0:
        merged = np.sum(np.stack(merged,axis=0),axis=0)
        merged = np.uint8(np.clip(merged, 0, 1)*255)
        res = Image.fromarray(merged)
        
        if name==None:
            res.save(os.path.join(save_path,'batch_0' + '_merged_road.jpg'))
        else:
            res.save(os.path.join(save_path,name + '_merged_road.jpg'))
    else:
        import pdb;pdb.set_trace()
        pass
    
    res_coef_list=[]
    res_interpolated_list=[]
    coef_all_roads = np.zeros((400,200,3),np.float32)
    for k in range(len(result['boxes'])):
        control = result['boxes'][k] 
        coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)#[2,2] 
        interpolated = bezier.interpolate_bezier(control,100)  
        res_interpolated_list.append(np.copy(interpolated))  
        # line = my_color_line_maker(interpolated,detected_endpoints[k],size=(196,200))###给占据的像素赋1.其中endpoint处膨胀
        line2 = my_color_line_maker(interpolated,coef_endpoints,size=(400,200))
        # res_list.append(line)
        res_coef_list.append(line2)
        coef_all_roads = coef_all_roads + np.float32(line2)
    
    if len(res_coef_list) > 0:
        all_lanes = np.zeros((400,200))
        for k in range(len(res_coef_list)):
            res_coef = Image.fromarray(res_coef_list[k])
            if name==None:
                res_coef.save(os.path.join(save_path,'batch_0' + '_est_coef_interp_road_'+str(k)+'.jpg'))
            else:
                res_coef.save(os.path.join(save_path,name + '_est_coef_interp_road_'+str(k)+'.jpg'))
                
        all_lanes = np.uint8(np.clip(all_lanes,0,1)*255)
        if name==None:
            coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
            temp_img = Image.fromarray(coef_all_roads)
            temp_img.save(os.path.join(save_path,'batch_0' + '_est_coef_all_roads.jpg' ))       
        else:
            coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
            temp_img = Image.fromarray(coef_all_roads)
            temp_img.save(os.path.join(save_path,name + '_est_coef_all_roads.jpg' ))    


    # #使用result里本来就有的interpolated_points和merged_interpolated_points画
    # res_coef_list=[]
    # coef_all_roads = np.zeros((400,200,3),np.float32)
    # for k in range(len(result['interpolated_points'])):
    #     interpolated = result['interpolated_points'][k]
    #     endpoints = np.concatenate([interpolated[0:1,:],interpolated[-1:,:]],axis=0)#[2,2] 
    #     line = my_color_line_maker(interpolated,endpoints,size=(400,200))
    #     res_coef_list.append(line)
    #     coef_all_roads = coef_all_roads + np.float32(line)
    
    # if len(res_coef_list) > 0:
    #     all_lanes = np.zeros((400,200))
    #     for k in range(len(res_coef_list)):
    #         res_coef = Image.fromarray(res_coef_list[k])
    #         if name==None:
    #             res_coef.save(os.path.join(save_path,'batch_0' + 'model_est_coef_interp_road_'+str(k)+'.jpg'))
    #         else:
    #             res_coef.save(os.path.join(save_path,name + 'model_est_coef_interp_road_'+str(k)+'.jpg'))
                
    #     all_lanes = np.uint8(np.clip(all_lanes,0,1)*255)
    #     if name==None:
    #         coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
    #         temp_img = Image.fromarray(coef_all_roads)
    #         temp_img.save(os.path.join(save_path,'batch_0' + 'model_est_coef_all_roads.jpg' ))       
    #     else:
    #         coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
    #         temp_img = Image.fromarray(coef_all_roads)
    #         temp_img.save(os.path.join(save_path,name + 'model_est_coef_all_roads.jpg' ))    
    
    # merged_lines = []
    # for k in range(len(result['merged_interpolated_points'])):
    #     merged_interpolated = result['merged_interpolated_points'][k]
    #     merged_endpoints = np.concatenate([merged_interpolated[0:1,:],merged_interpolated[-1:,:]],axis=0)
    #     line = np.float32(my_color_line_maker(merged_interpolated,merged_endpoints,size=(400,200)))/255
    #     merged_lines.append(line)  
        
    # merged_lines = np.sum(np.stack(merged_lines,axis=0),axis=0)
    # merged_lines = np.uint8(np.clip(merged_lines, 0, 1)*255)
    # res = Image.fromarray(merged_lines)
    # if name==None:
    #     res.save(os.path.join(save_path,'batch_0' + '_model_merged_road.jpg'))
    # else:
    #     res.save(os.path.join(save_path,name + '_model_merged_road.jpg'))
        

def vis_target(target,save_path,name=None):
    
    img_centers = target['center_img']
    img_centers = img_centers.cpu().numpy().squeeze(0)
    
    roads = target['roads'].cpu().numpy().squeeze(0)

    orig_coefs = target['control_points'].cpu().numpy().squeeze(0)
    coef_endpoints = get_endpoints_from_coeffs(orig_coefs)
    
    coef_all_roads = np.zeros((img_centers.shape[0],img_centers.shape[1],3))
    
    for k in range(len(roads)):
        #20231105:之后可以试一下不加端点
        cur_coef_full = add_endpoints_to_line(np.float32(img_centers == roads[k]),coef_endpoints[k])#(196, 200, 3)
        
        temp_coef_img = Image.fromarray(np.uint8(cur_coef_full*255))
        
        coef_all_roads = coef_all_roads + cur_coef_full

        if name==None:                
                temp_coef_img.save(os.path.join(save_path,'batch_0'+ '_gt_coef_visible_road_'+str(k)+'.jpg' ))
        else:
                temp_coef_img.save(os.path.join(save_path,name + '_gt_coef_visible_road_'+str(k)+'.jpg' ))

    coef_all_roads = np.clip(coef_all_roads,0,1)

    if name==None:
        temp_coef_img = Image.fromarray(np.uint8(coef_all_roads*255))
        temp_coef_img.save(os.path.join(save_path,'batch_0' + '_gt_coef_visible_all_roads.jpg' ))
        
    else:
        temp_coef_img = Image.fromarray(np.uint8(coef_all_roads*255))
        temp_coef_img.save(os.path.join(save_path,name + '_gt_coef_visible_all_roads.jpg' ))

    merged = get_target_merged_network(target)
    if len(merged) > 0:
        merged = np.sum(np.stack(merged,axis=0),axis=0)
        merged = np.uint8(np.clip(merged, 0, 1)*255)
        res = Image.fromarray(merged)
        
        if name==None:
            res.save(os.path.join(save_path,'batch_0' + '_merged_road.jpg'))
        else:
            res.save(os.path.join(save_path,name + 'gt_merged_road.jpg'))
    else:
        import pdb;pdb.set_trace()
        pass

def vis_results_eval(results, target):
    base_path = os.path.join('/mnt/data/zty/show_dir','val_images','scene102-104','work_dir20231225')
    os.makedirs(base_path,exist_ok=True)
    fileList = glob.glob(os.path.join(base_path,'*'))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    
    result = results[0]
    vis_target(target,base_path,name='_')
    vis_est(result,base_path,name='_')


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
    # import pdb;pdb.set_trace()
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

    img_norm_cfg = cfg.img_norm_cfg

    # get denormalized param
    mean = np.array(img_norm_cfg['mean'],dtype=np.float32)
    std = np.array(img_norm_cfg['std'],dtype=np.float32)
    to_bgr = img_norm_cfg['to_rgb']

    # get pc_range
    pc_range = cfg.point_cloud_range

    # get car icon
    car_img = Image.open('./figs/lidar_car.png')

    # get color map: divider->r, ped->b, boundary->g
    colors_plt = ['r', 'b', 'g']


    logger.info('BEGIN vis test dataset samples gt label & pred')

    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    have_mask = False
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    # a = dataset[0]
    # import pdb;pdb.set_trace()
    
    for i, data in enumerate(data_loader):
        # ###
        # if i!=41:
        #     continue
        # ###
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            prog_bar.update()  
            continue
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
    
        img = data['img'].data[0][0][-1] #torch.Size([6, 3, 480, 800])
        img_metas = data['img_metas'].data[0][0][1]#是当前时刻的img_metas
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0][0] #LiDARInstanceLines
        gt_labels_3d = data['gt_labels_3d'].data[0][0] #torch.Size([6])

        pts_filename = img_metas['pts_filename']
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]
        sample_dir = osp.join(args.show_dir, pts_filename)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))


        ###vis stsu
        #target.keys() = dict_keys(['center_img', 'labels', 'roads', 'control_points', 'con_matrix'])
        target = data['target']
        static_thresh = 0.5
        assoc_thresh = 0.75

        vis_results_eval(result, target) 
        

        import pdb;pdb.set_trace()

        for vis_format in args.gt_format:
            if vis_format == 'fixed_num_pts':
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')
                # gt_bboxes_3d[0].fixed_num=30 #TODO, this is a hack
                gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
                    # import pdb;pdb.set_trace() 
                    pts = gt_bbox_3d.numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])
                    # plt.plot(x, y, color=colors_plt[gt_label_3d])
                    # plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                map_path = osp.join(sample_dir, 'GT_fixednum_pts_MAP.jpg')
                plt.savefig(map_path, bbox_inches='tight', dpi=400)
                plt.close()         
            else: 
                logger.error(f'WRONG visformat for GT: {vis_format}')
                raise ValueError(f'WRONG visformat for GT: {vis_format}')



        # import pdb;pdb.set_trace()
        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')

        # visualize pred
        # import pdb;pdb.set_trace()
        result_dic = result[0]['pts_bbox']
        boxes_3d = result_dic['boxes_3d'] # bbox: xmin, ymin, xmax, ymax
        scores_3d = result_dic['scores_3d']
        labels_3d = result_dic['labels_3d']
        pts_3d = result_dic['pts_3d']
        for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d, boxes_3d,labels_3d, pts_3d):
            pred_pts_3d = pred_pts_3d.numpy()
            pts_x = pred_pts_3d[:,0]
            pts_y = pred_pts_3d[:,1]
            # plt.plot(pts_x, pts_x, color=colors_plt[gt_label_3d])
            # plt.scatter(pts_x, pts_y, s=1,color=colors_plt[pred_label_3d])
            plt.quiver(pts_x[:-1], pts_y[:-1], pts_x[1:] - pts_x[:-1], pts_y[1:] - pts_y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[pred_label_3d])
            pred_bbox_3d = pred_bbox_3d.numpy()
            xy = (pred_bbox_3d[0],pred_bbox_3d[1])
            width = pred_bbox_3d[2] - pred_bbox_3d[0]
            height = pred_bbox_3d[3] - pred_bbox_3d[1]
            pred_score_3d = float(pred_score_3d)
            pred_score_3d = round(pred_score_3d, 2)
            s = str(pred_score_3d)
            plt.text(pts_x[0], pts_y[0], s,  fontsize=2)

        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        map_path = osp.join(sample_dir, 'PRED_MAP.jpg')
        plt.savefig(map_path, bbox_inches='tight', dpi=400)
        plt.close()
        
        
        # import pdb;pdb.set_trace()
        prog_bar.update()

    logger.info('\n DONE vis test dataset samples gt label & pred')
if __name__ == '__main__':
    main()
