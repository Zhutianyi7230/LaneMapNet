import numpy as np
import scipy.ndimage as ndimage

def get_vertices(adj):

    ins = []
    outs = []
    for k in range(len(adj)):
    #for k in range(7):
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

def get_merged_coeffs(targets):
    coeffs = targets['boxes']
    assoc = targets['assoc'] 
    diag_mask = np.eye(len(assoc))
    
    diag_mask = 1 - diag_mask
    assoc = assoc*diag_mask#过滤掉对角线
    
    corrected_coeffs = np.copy(coeffs)
    ins, outs = get_vertices(assoc)
    
    for k in range(len(ins)):
        all_points=[]
        #对于在ins中的line，添加末端点；对于在outs中的line，添加起始点
        for m in ins[k]:
            all_points.append(corrected_coeffs[m,-1])
            
        for m in outs[k]:
            all_points.append(corrected_coeffs[m,0])
            
        av_p = np.mean(np.stack(all_points,axis=0),axis=0)#中点
        
        #相当于对于要连接的两条line，出发line的末端点和到达line的起始点都往中点处延申了
        for m in ins[k]:
            corrected_coeffs[m,-1] = av_p
            
        for m in outs[k]:
            corrected_coeffs[m,0] = av_p
    
    return corrected_coeffs

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
    # struct = ndimage.generate_binary_structure(5, 2)
    
    # logging.error('STRUCT ' + str(struct))
    # logging.error('BASE START ' + str(base_start.shape))
    
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