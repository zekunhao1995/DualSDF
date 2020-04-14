import torch
import torch.nn as nn

class BaseTrainer(nn.Module):
    def __init__(self, cfg, args, device):
        self.cfg = cfg
        self.args = args
        self.device = device
