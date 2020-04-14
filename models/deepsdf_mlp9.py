import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.dropout = cfg.dropout
        dropout_prob = cfg.dropout_prob
        self.use_tanh = cfg.use_tanh
        in_ch = cfg.in_ch
        out_ch = cfg.out_ch
        feat_ch = cfg.hidden_ch
        
        print("[DeepSDF MLP-9] Dropout: {}; Do_prob: {}; in_ch: {}; hidden_ch: {}".format(self.dropout, dropout_prob, in_ch, feat_ch))
        if self.dropout is False:
            self.net1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
                nn.ReLU(inplace=True)
            )

            self.net2 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Linear(feat_ch, out_ch)
            )
        else:
            self.net1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
            )

            self.net2 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.Linear(feat_ch, out_ch)
            )

        num_params = sum(p.numel() for p in self.parameters())
        print('[num parameters: {}]'.format(num_params))

    # z: [B 131] 128-dim latent code + 3-dim xyz
    def forward(self, z):
        #bs = z.size(0)
        #N = p.size(1)
        #if len(z.shape) == 2:
        #    z = z.unsqueeze(-2).expand(-1,N,-1)
        #in1 = torch.cat([z, p], dim=-1)
        in1 = z
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)
        out2 = self.net2(in2)
        if self.use_tanh:
            out2 = torch.tanh(out2)
        return out2
 
