# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time
import os

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util
from projects.mmdet3d_plugin.datasets import bezier
from projects.mmdet3d_plugin.maptr.detectors.utils import BinaryConfusionMatrix

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    
    static_threshs = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    # static_threshs = [0.5,0.6,0.7,0.8,0.9]
    confusions = [BinaryConfusionMatrix(static_thresh) for static_thresh in static_threshs]
    mean_recalls = []
    mean_pres = []
    mean_fs = []
    mean_assoc_recalls = []
    mean_assoc_pres = []
    mean_assoc_fs = []

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True   
            else:
                batch_size = len(result)
                bbox_results.extend(result)

            #if isinstance(result[0], tuple):
            #    assert False, 'this code is for instance segmentation, which our code will not utilize.'
            #    result = [(bbox_results, encode_mask_results(mask_results))
            #              for bbox_results, mask_results in result]

            target = data['target']
            #验证Lane Network Recognization指标
            for t in range(len(static_threshs)):
                static_thresh = static_threshs[t]
                confusion = confusions[t]
                #加阈值筛选
                thresh_select(result[0]['pts_stsu'], static_thresh)             
                
                hausdorff_static_dist, hausdorff_static_idx, hausdorff_gt = hausdorff_match(result[0]['pts_stsu'], target)
                confusion.update(result[0]['pts_stsu'], hausdorff_gt, hausdorff_static_idx, target)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
    file1 = open(os.path.join('/home/ps/ztyhdmap/MapTR13/LaneMapNet/work_dir20231225','val_res_train' + '_0.05.txt'),"a")
    for t in range(len(static_threshs)):
        confusion = confusions[t]
        #获得confusion指标结果
        static_res_dict = confusion.get_res_dict
        file1.write('--------------------------------'+str(static_threshs[t])+'--------------------------------\n')
        for k in static_res_dict.keys():
            file1.write(str(k) + ' : ' + str(static_res_dict[k]) + ' \n')
    
        mean_recalls.append(static_res_dict['mean_recall'])
        mean_pres.append(static_res_dict['mean_pre'])
        mean_fs.append(static_res_dict['mean_f_score'])
        mean_assoc_recalls.append(static_res_dict['mean_assoc_recall'])
        mean_assoc_pres.append(static_res_dict['mean_assoc_pre'])
        mean_assoc_fs.append(static_res_dict['mean_assoc_f'])
    
    file1.write('\n')
    file1.write('f_score' + " : " + str(np.mean(mean_fs)) + '\n')
    file1.write('assoc_f_score' + " : " + str(np.mean(mean_assoc_fs)) + '\n')
    file1.write('-----------------------------------------------------')
    file1.write('\n')
    file1.write('\n')
    file1.close()    

    # collect results from all ranks
    if gpu_collect:#False
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if mask_results is None:
        return bbox_results
        #bbox_results[0].keys() = dict_keys(['pts_bbox', 'pts_stsu'])
    return {'bbox_results': bbox_results, 'mask_results': mask_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)


def hausdorff_match(out,target):
    est_coefs = out['boxes']
    
    orig_coefs = target['control_points'].cpu().numpy()
    orig_coefs = np.reshape(orig_coefs, (-1, int(orig_coefs.shape[-1]/2),2))
    
    interpolated_origs = []
    
    for k in range(len(orig_coefs)):
        inter = bezier.interpolate_bezier(orig_coefs[k],100)
        interpolated_origs.append(np.copy(inter))
        
    if len(est_coefs) == 0:
        return None,[], interpolated_origs
    dist_mat = np.mean(np.sum(np.square(np.expand_dims(est_coefs,axis=1) - np.expand_dims(orig_coefs,axis=0)),axis=-1),axis=-1)#(预测数量，GT数量)
    
    ind = np.argmin(dist_mat, axis=-1)#(预测数量,),表示每个预测最接近的GT序号
    min_vals = np.min(dist_mat,axis=-1)
    return min_vals, ind, interpolated_origs 

def thresh_select(out, static_thresh):
    if len(out['scores'])!=0:
        selected = (out['scores']>static_thresh)
        for key in ['labels','boxes','scores','interpolated_points', 'merged_interpolated_points']:
            out[key] = out[key][selected]
        out['assoc'] = out['assoc'][selected,:]
        out['assoc'] = out['assoc'][:,selected]

        


    
