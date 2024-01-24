import torch
import logging
import numpy as np
import cv2
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
#from mean_average_precision import MeanAveragePrecision
def render_polygon(mask, polygon, shape, value=1):
    
    to_mult = np.expand_dims(np.array([shape[1],shape[0]]),axis=0)
    polygon = polygon*to_mult
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)


class BinaryConfusionMatrix(object):

    def __init__(self, static_thresh):
        
        self.static_steps = [0.01,  0.02, 0.03 , 0.04, 0.05, 0.06, 0.07, 0.08 ,0.09 ,0.1]

        self.matched_gt = 0
        self.unmatched_gt = 0
        
        self.merged_matched_gt = 0
        self.merged_unmatched_gt = 0
     
#       
        self.static_pr_total_est = 0
        self.static_pr_total_gt = 0
        self.static_pr_tp = []
        self.static_pr_fn = []
        self.static_pr_fp = []

        for k in range(len(self.static_steps)):
            self.static_pr_tp.append(0)
            self.static_pr_fn.append(0)
            self.static_pr_fp.append(0)
            
            
        self.merged_static_pr_total_est = 0
        self.merged_static_pr_total_gt = 0
        self.merged_static_pr_tp = []
        self.merged_static_pr_fn = []
        self.merged_static_pr_fp = []

        for k in range(len(self.static_steps)):
            self.merged_static_pr_tp.append(0)
            self.merged_static_pr_fn.append(0)
            self.merged_static_pr_fp.append(0)
        
        self.assoc_threshs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]        

        self.assoc_tps = np.zeros(len(self.assoc_threshs))
        self.assoc_fns = np.zeros(len(self.assoc_threshs))
        self.assoc_fps = np.zeros(len(self.assoc_threshs))
        
        self.static_mse_list=[]
        self.static_metrics = dict()
        
        self.scene_name = ''
        self.ap_list=np.zeros((8))


    def update(self, out, haus_gt, haus_idx, targets):
        '''
        PRECISION-RECALL
        '''
        res_interpolated_list = out['interpolated_points']

        num_estimates = len(res_interpolated_list)
        num_gt = len(haus_gt)

        if num_estimates == 0:
            for k in range(len(self.static_steps)):
            
                self.static_pr_fn[k] = np.copy(self.static_pr_fn[k]) + len(haus_gt)*len(haus_gt[0]) 
                
            self.unmatched_gt += len(haus_gt)
        
        else:
            
            '''
            PRE STATIC
            '''
            
            m_g = len(np.unique(np.array(haus_idx)))#有被预测匹配到的gt个数
            self.matched_gt += m_g
            self.unmatched_gt += len(haus_gt) - m_g
            
            for est_id in range(num_estimates):
                cur_gt = haus_gt[haus_idx[est_id]]
                
                dis = cdist(res_interpolated_list[est_id],cur_gt,'euclidean')

                res_dis = np.min(dis,axis=-1)
                
                self.static_pr_total_gt += len(cur_gt)
                self.static_pr_total_est += len(res_interpolated_list[est_id])
                
                for k in range(len(self.static_steps)):
                
                    self.static_pr_tp[k] = np.copy(self.static_pr_tp[k]) + np.sum(res_dis < self.static_steps[k]) 
                    self.static_pr_fp[k] = np.copy(self.static_pr_fp[k]) + np.sum(res_dis > self.static_steps[k]) 

                    # print(self.static_pr_tp[k],self.static_pr_fp[k])
                    
                
            for gt_id in range(len(haus_gt)):
                m = np.ones([100])
                cur_est_ids = np.where(haus_idx==gt_id)[0]
                if len(cur_est_ids) == 0 :
                    for k in range(len(self.static_steps)):
                        self.static_pr_fn[k] = np.copy(self.static_pr_fn[k]) + 100
                else:
                    for k in range(len(self.static_steps)):
                        for cur_est_id in cur_est_ids:
                            dis = cdist(res_interpolated_list[cur_est_id],haus_gt[gt_id],'euclidean')
                            gt_dis = np.min(dis,axis=0)
                            mask = np.where(gt_dis < self.static_steps[k])[0]
                            m[mask] = 0

                        self.static_pr_fn[k] = np.copy(self.static_pr_fn[k]) + np.sum(m)
#                       
#          
        '''
        ASSOC LOSS
        '''
        for i in range(len(self.assoc_threshs)):
            assoc_thresh = self.assoc_threshs[i]
            assoc_tp = 0
            assoc_fp = 0
            assoc_fn = 0
            gt_con_matrix = targets['con_matrix'][0].cpu().numpy()
            if len(haus_idx)!=0:      
                assoc_est = out['assoc'].cpu().numpy()
                ##先计算tp,fp
                for est_id in range(num_estimates):
                    matched_gt = haus_idx[est_id]
                    cur_gt_assoc = gt_con_matrix[matched_gt]
                    cur_est_assoc = assoc_est[est_id]
            
                    for m in range(len(cur_est_assoc)):
                        if cur_est_assoc[m] > assoc_thresh:
                            temp_id = haus_idx[m]
                            if temp_id == matched_gt:
                                assoc_tp += 1
                            elif cur_gt_assoc[temp_id] > 0.5: 
                                assoc_tp += 1
                            else:
                                assoc_fp += 1

            ##再计算fn  
            for gt_id in range(len(gt_con_matrix)):
                cur_gt_assoc = gt_con_matrix[gt_id]
                
                temp_mat = np.copy(cur_gt_assoc)
                temp_mat = -temp_mat

                # if not np.any(haus_idx == None):
                if not len(haus_idx)==0:
                    
                    if gt_id in haus_idx:
                        matched_ests = np.where(np.array(haus_idx)==gt_id)[0]
                        
                        for m in range(len(cur_gt_assoc)):
                        
                            if cur_gt_assoc[m] > 0.5:
                                
                                if temp_mat[m] == -1:
                                    
                                    
                                    if m in haus_idx:
                                        other_ests = np.where(np.array(haus_idx)==m)[0]
                                            
                                        cur_est_assoc = assoc_est[matched_ests]
    #                                        temp_found = False
                                        for my_est in range(len(cur_est_assoc)):
                                            if np.any(cur_est_assoc[my_est][other_ests] > assoc_thresh):
    #                                                temp_found=True
                                                temp_mat[m] = 1
                                                break
                                                    
                        assoc_fn += np.sum(temp_mat == -1)
                                        
                    else:
                        assoc_fn += np.sum(cur_gt_assoc)
                else:
                    assoc_fn += np.sum(cur_gt_assoc)
            
            # print(assoc_thresh, assoc_fn,assoc_fp,assoc_tp)
            self.assoc_fns[i] += assoc_fn
            self.assoc_fps[i] += assoc_fp
            self.assoc_tps[i] += assoc_tp
        
            
    @property
    def get_res_dict(self):
 
        rec_list = []
        pre_list = []
        for k in range(len(self.static_steps)):
            
            self.static_metrics['precision_'+str(self.static_steps[k])] = self.static_pr_tp[k]/(self.static_pr_fp[k] + self.static_pr_tp[k] + 0.0001)
            self.static_metrics['recall_'+str(self.static_steps[k])] = self.static_pr_tp[k]/(self.static_pr_fn[k] + self.static_pr_tp[k] + 0.0001)
            pre_list.append(self.static_pr_tp[k]/(self.static_pr_fp[k] + self.static_pr_tp[k] + 0.0001))
            rec_list.append(self.static_pr_tp[k]/(self.static_pr_fn[k] + self.static_pr_tp[k] + 0.0001))
            
        self.static_metrics['mean_recall'] = np.mean(rec_list)
        self.static_metrics['mean_pre'] = np.mean(pre_list)
        
        
        self.static_metrics['mean_f_score'] = np.mean(pre_list)*np.mean(rec_list)*2/(np.mean(pre_list)+np.mean(rec_list)+ 0.001)

        
        self.static_metrics['mse'] = np.mean(self.static_mse_list)
  
        # self.static_metrics['assoc_iou'] = self.assoc_tps/(self.assoc_tps + self.assoc_fns + self.assoc_fps + 0.0001)
        
        assoc_rec_list = []
        assoc_pre_list = []
        for i in range(len(self.assoc_threshs)):
            self.static_metrics['assoc_precision_'+str(self.assoc_threshs[i])] = self.assoc_tps[i]/(self.assoc_tps[i] + self.assoc_fps[i] +  0.0001)
            self.static_metrics['assoc_recall_'+str(self.assoc_threshs[i])] = self.assoc_tps[i]/(self.assoc_tps[i] + self.assoc_fns[i] +  0.0001)
            assoc_pre_list.append(self.assoc_tps[i]/(self.assoc_tps[i] + self.assoc_fps[i] +  0.0001))
            assoc_rec_list.append(self.assoc_tps[i]/(self.assoc_tps[i] + self.assoc_fns[i] +  0.0001))
            
        self.static_metrics['mean_assoc_recall'] = np.mean(assoc_rec_list)
        self.static_metrics['mean_assoc_pre'] = np.mean(assoc_pre_list)
        
        self.static_metrics['mean_assoc_f'] = self.static_metrics['mean_assoc_pre']*self.static_metrics['mean_assoc_recall']*2/(self.static_metrics['mean_assoc_pre']+self.static_metrics['mean_assoc_recall']+ 0.001)
        
        self.static_metrics['matched_gt'] = self.matched_gt
        self.static_metrics['unmatched_gt'] = self.unmatched_gt
        self.static_metrics['detection_ratio'] = self.matched_gt/(self.matched_gt+self.unmatched_gt+ 0.001)
          
        return self.static_metrics
        
                
    @property
    def static_mse(self):
#        return self.tp.float() / (self.tp + self.fn + self.fp).float()
        return np.mean(self.static_mse_list)
    
    
    def reset(self):
        
        
        self.matched_gt = 0
        self.unmatched_gt = 0
        
        self.merged_matched_gt = 0
        self.merged_unmatched_gt = 0

#       
        self.static_pr_total_est = 0
        self.static_pr_total_gt = 0
        self.static_pr_tp = []
        self.static_pr_fn = []
        self.static_pr_fp = []

        for k in range(len(self.static_steps)):
            self.static_pr_tp.append(0)
            self.static_pr_fn.append(0)
            self.static_pr_fp.append(0)
            
            
        self.merged_static_pr_total_est = 0
        self.merged_static_pr_total_gt = 0
        self.merged_static_pr_tp = []
        self.merged_static_pr_fn = []
        self.merged_static_pr_fp = []

        for k in range(len(self.static_steps)):
            self.merged_static_pr_tp.append(0)
            self.merged_static_pr_fn.append(0)
            self.merged_static_pr_fp.append(0)
        
        self.assoc_tp = 0
        self.assoc_fn = 0
        self.assoc_fp = 0
      
        self.static_mse_list=[]
        
        self.static_metrics = dict()
        
        self.scene_name = ''
        
        self.ap_list=np.zeros((8))

    
    @property
    def mean_iou(self):
        # Only compute mean over classes with at least one ground truth
        valid = (self.tp + self.fn) > 0
        if not valid.any():
            return 0
        return float(self.iou[valid].mean())

    @property
    def dice(self):
        return 2 * self.tp.float() / (2 * self.tp + self.fp + self.fn).float()
    
    @property
    def macro_dice(self):
        valid = (self.tp + self.fn) > 0
        if not valid.any():
            return 0
        return float(self.dice[valid].mean())
    
    @property
    def precision(self):
        return self.tp.float() / (self.tp + self.fp).float()
    
    @property
    def recall(self):
        return self.tp.float() / (self.tp + self.fn).float()