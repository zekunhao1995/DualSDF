import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Original DeepSDF loss
def clamped_l1(pred_dist, gt_dist, trunc=0.1):
    pred_dist_trunc = torch.clamp(pred_dist, -trunc, trunc)
    gt_dist_trunc = torch.clamp(gt_dist, -trunc, trunc)
    loss = torch.abs(pred_dist_trunc - gt_dist_trunc)
    return loss

# [B N]
def clamped_l1_correct(pred_dist, gt_dist, trunc=0.1):
    pred_dist_lower = torch.clamp(pred_dist, None, trunc)
    pred_dist_upper = torch.clamp(pred_dist, -trunc, None)
    pos_trunced_mask = (gt_dist >= trunc)
    neg_trunced_mask = (gt_dist <= -trunc)
    valid_mask = ~(pos_trunced_mask|neg_trunced_mask)      
    
    loss_valid = torch.sum(torch.abs(pred_dist - gt_dist) * valid_mask.float(), dim=-1)
    loss_lower = torch.sum((trunc - pred_dist_lower) * pos_trunced_mask.float(), dim=-1)
    loss_upper = torch.sum((pred_dist_upper + trunc) * neg_trunced_mask.float(), dim=-1)
    loss = (loss_lower + loss_upper + loss_valid) / pred_dist.size(1)
    return loss
    
# L2 loss on the outside, encourage inside to < 0.0
def onesided_l2(pred_dist, gt_dist):
    valid_mask = (gt_dist >= 0.0).float()
    num_valid = torch.sum(valid_mask, dim=-1)
    num_inside = valid_mask[0].numel() - num_valid
    loss_valid = torch.sum((gt_dist-pred_dist)**2 * valid_mask, dim=-1) / (num_valid+1e-8)
    loss_inside = torch.sum(torch.clamp(pred_dist, 0.0, None) * (1.0-valid_mask), dim=-1) / (num_inside+1e-8)
    
    loss = loss_valid + loss_inside
    return loss
