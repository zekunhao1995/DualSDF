import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k*a, dim=dim, keepdim=keepdim) / k
    return dmix

class SDFFun(nn.Module):
    def __init__(self, cfg):
        super(SDFFun, self).__init__()
        self.return_idx = cfg.return_idx
        self.smooth = cfg.smooth
        self.smooth_factor = cfg.smooth_factor
        print('[SdfSphere] return idx: {}; smooth: {}'.format(self.return_idx, self.smooth))
        
    # assume we use Sphere primitive for everything
    # parameters: radius[r], center[xyz]
    def prim_sphere_batched_smooth(self, x, p):
        device = x.device
        x = x.unsqueeze(-2) # B N 1 3
        p = p.unsqueeze(-3) # B 1 M 4
        logr = p[:,:,:,0]
        d = torch.sqrt(torch.sum((x-p[:,:,:,1:4])**2, dim=-1)) - torch.exp(logr) # B N M 
        if self.return_idx:
            d, loc = torch.min(d, dim=-1, keepdim=True)
            return d, loc
        else:
            if self.smooth:
                d = bsmin(d, dim=-1, k=self.smooth_factor, keepdim=True)
            else:
                d, _ = torch.min(d, dim=-1, keepdim=True)
            return d

    # a: [B M 4]; 
    # x: [B N 3]; x, y, z \in [-0.5, 0.5]
    def forward(self, a, x):
        a = a.reshape(a.size(0), -1, 4)
        out = self.prim_sphere_batched_smooth(x, a)
        return out
