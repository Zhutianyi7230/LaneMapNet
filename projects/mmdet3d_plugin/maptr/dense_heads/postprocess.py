import torch
from torch import nn
import torch.nn.functional as F



class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox,  out_assoc = outputs['pred_logits'], outputs['pred_spline'],  outputs['pred_assoc']
     
        # assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)

        est = torch.reshape(out_bbox,(len(out_bbox),out_bbox.shape[1],-1,2))#[1, 100, 3, 2]
        results = [{'scores': s, 'labels': l, 'boxes': b,'probs': p,'assoc': a} for s, l, b, p, a in zip(scores, labels, est,prob,out_assoc)]
        
        return results
        

def build_postprocess():
    return PostProcess()