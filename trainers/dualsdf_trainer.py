import os
import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import itertools

from trainers.base_trainer import BaseTrainer
import toolbox.lr_scheduler
import models.embeddings

def KLD(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    KLD = torch.mean(KLD)
    return KLD
    
class Trainer(BaseTrainer):
    def __init__(self, cfg, args, device):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.device = device

        # Init models
        deepsdf_lib = importlib.import_module(cfg.models.deepsdf.type)
        self.deepsdf_net = deepsdf_lib.Decoder(cfg.models.deepsdf)
        self.deepsdf_net.to(self.device)
        print("DeepSDF Net:")
        print(self.deepsdf_net)
        
        prim_attr_lib = importlib.import_module(cfg.models.prim_attr.type)
        self.prim_attr_net = prim_attr_lib.Decoder(cfg.models.prim_attr)
        self.prim_attr_net.to(self.device)
        print("Prim Attr Net:")
        print(self.prim_attr_net)
        
        prim_sdf_lib = importlib.import_module(cfg.models.prim_sdf.type)
        self.prim_sdf_fun = prim_sdf_lib.SDFFun(cfg.models.prim_sdf)
        self.prim_sdf_fun.to(self.device)
        print("Prim SDF Fun:")
        print(self.prim_sdf_fun)
        
        # Init loss functions
        self.lossfun_fine = self._get_lossfun(self.cfg.trainer.loss_fine)
        self.lossfun_coarse = self._get_lossfun(self.cfg.trainer.loss_coarse)
        
        # Init optimizers
        self.optim_deepsdf, self.lrscheduler_deepsdf = self._get_optim(self.deepsdf_net.parameters(), self.cfg.trainer.optim_deepsdf)
        self.optim_primitive, self.lrscheduler_primitive = self._get_optim(self.prim_attr_net.parameters(), self.cfg.trainer.optim_primitive)
        
        self.additional_log_info = {}
    
    # Init training-specific contexts
    def prep_train(self):
        self.sid2idx = {k:v for v, k in enumerate(sorted(self.cfg.train_shape_ids))}
        print('[DualSDF Trainer] init. #entries in sid2idx: {}'.format(len(self.sid2idx)))
        # Init latent code
        self.latent_embeddings = self._get_latent(self.cfg.trainer.latent_code, N=len(self.sid2idx))
        self.optim_latentcode, self.lrscheduler_latentcode = self._get_optim(self.latent_embeddings.parameters(), self.cfg.trainer.optim_latentcode)
        self.train()
        
    def _get_latent(self, cfg, N):
        embedding = getattr(models.embeddings, cfg.type)
        embedding_instance = embedding(cfg, N=N, dim=self.cfg.trainer.latent_dim).to(self.device)
        return embedding_instance
        
    def _get_optim(self, parameters, cfg):
        if cfg.type.lower() == "adam":
            optim = torch.optim.Adam(parameters, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay, amsgrad=False)
        elif cfg.type.lower() == "sgd":
            optim = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        else:
            raise NotImplementedError("Unknow optimizer: {}".format(cfg.type))
        
        scheduler = None
        if hasattr(cfg, 'lr_scheduler'):
            scheduler = getattr(toolbox.lr_scheduler, cfg.lr_scheduler.type)(cfg.lr_scheduler)
        return optim, scheduler
    
    def _step_lr(self, epoch):
        lr_latentcode = self.lrscheduler_latentcode(epoch)
        for g in self.optim_latentcode.param_groups:
            g['lr'] = lr_latentcode
        lr_deepsdf = self.lrscheduler_deepsdf(epoch)
        for g in self.optim_deepsdf.param_groups:
            g['lr'] = lr_deepsdf
        lr_primitive = self.lrscheduler_primitive(epoch)
        for g in self.optim_primitive.param_groups:
            g['lr'] = lr_primitive
        print('Step LR: L: {}; D: {}; P: {}'.format(lr_latentcode, lr_deepsdf, lr_primitive))

    def _get_lossfun(self, cfg):
        print(cfg)
        if cfg.type.lower() == 'clamped_l1':
            from models.lossfuns import clamped_l1
            lossfun = lambda pred, gt: torch.mean(clamped_l1(pred, gt, trunc=cfg.trunc), dim=-1)
        elif cfg.type.lower() == 'clamped_l1_correct':
            from models.lossfuns import clamped_l1_correct as clamped_l1
            lossfun = lambda pred, gt: clamped_l1(pred, gt, trunc=cfg.trunc)
        elif cfg.type.lower() == 'l1':
            lossfun = lambda pred, gt: torch.mean(torch.abs(pred-gt), dim=-1)
        elif cfg.type.lower() == 'onesided_l2':
            from models.lossfuns import onesided_l2
            lossfun = onesided_l2
        else:
            raise NotImplementedError("Unknow loss function: {}".format(cfg.type))
        return lossfun
        
    # loss?: [B]
    def _reduce_loss(self, loss1, loss2):
        if self.cfg.trainer.mixture_loss:
            loss_s = torch.stack([loss1, loss2], dim=-1)
            loss = torch.mean(torch.logsumexp(loss_s, dim=-1)) - np.log(2)
        else:
            loss = 0.5 * (torch.mean(loss1 + loss2))
        return loss
        
    def _b_idx2latent(self, latent_embeddings, indices, num_augment_pts=None):
        batch_latent_dict = latent_embeddings(indices, num_augment_pts=num_augment_pts)
        batch_latent = batch_latent_dict['latent_code']
        if 'mu' in batch_latent_dict.keys() and 'logvar' in batch_latent_dict.keys():
            batch_mu = batch_latent_dict['mu']
            batch_logvar = batch_latent_dict['logvar']
            kld = KLD(batch_mu, batch_logvar)
            self.additional_log_info['vad_batch_mu_std'] = torch.std(batch_mu).item()
            self.additional_log_info['vad_batch_kld'] = kld.item()
            if 'std' in batch_latent_dict.keys():
                batch_sigma = batch_latent_dict['std']
            else:
                batch_sigma = torch.exp(0.5*batch_logvar)
            self.additional_log_info['vad_batch_sigma_mean'] = torch.mean(batch_sigma).item()
        else:
            kld = 0.0
        
        if 'latent_code_augment' in batch_latent_dict.keys():
            batch_latent_aug = batch_latent_dict['latent_code_augment']
        else:
            batch_latent_aug = batch_latent
        return batch_latent, batch_latent_aug, kld
    
    # Convert list of shape ids to their corresponding indices in embedding.
    def _b_sid2idx(self, sid_list):
        data_indices = torch.tensor([self.sid2idx[x] for x in sid_list], dtype=torch.long, device=self.device)
        return data_indices
        
    # Z: [B, 128] or
    #    [B, N, 128]
    # P: [B, N, 3]
    def _forward_deepsdf(self, z, p):
        bs = z.size(0)
        N = p.size(1)
        if len(z.shape) == 2:
            z = z.unsqueeze(1).expand(-1,N,-1)
        inp = torch.cat([z, p], dim=-1)
        dists = self.deepsdf_net(inp) # [64 2048 1]
        return dists
    
    # Z: [B, 128]
    # P: [B, N, 3]
    def _forward_primitive(self, z, p):
        bs = z.size(0)
        N = p.size(1)
        attrs = self.prim_attr_net(z)
        dists = self.prim_sdf_fun(attrs, p)
        return dists, attrs
    
    def _reg_attr(self, attrs):
        attrs = attrs.reshape(attrs.size(0), -1, 4) # [B N rxyz]
        dists = torch.sum(attrs[:,:,1:]**2, dim=-1, keepdim= True)
        dists = torch.clamp(dists, 1.05, None)
        loss = torch.sum(dists - 1.05)
        return loss
        
    def epoch_start(self, epoch):
        # Setting LR
        self.train()
        self._step_lr(epoch)
        self.optim_latentcode.zero_grad()
        
    def step(self, data):
        data_f = data['surface_samples'].to(self.device, non_blocking=True) # [64 2048 4] xyzd
        data_c = data['sphere_samples'].to(self.device, non_blocking=True)
        data_indices = data['shape_indices'].squeeze(-1).to(self.device, non_blocking=True) # [64]
        data_ids = data['shape_ids']
        
        latent_codes_coarse, latent_codes_fine, kld = self._b_idx2latent(self.latent_embeddings, data_indices, num_augment_pts=data_f.size(1)) # [64 128]
        
        if self.cfg.trainer.detach_latent_coarse:
            latent_codes_coarse = latent_codes_coarse.detach()
        if self.cfg.trainer.detach_latent_fine:
            latent_codes_fine = latent_codes_fine.detach()

        self.optim_deepsdf.zero_grad()
        self.optim_primitive.zero_grad()
        
        # DeepSDF
        pts_fine = data_f[...,:3]
        dists_gt_fine = data_f[...,[3]].squeeze(-1)
        dists_deepsdf = self._forward_deepsdf(latent_codes_fine, pts_fine).squeeze(-1) # 64, 2048, 1
        
        # PrimitiveSDF
        pts_coarse = data_c[...,:3]
        dists_gt_coarse = data_c[...,[3]].squeeze(-1)
        dists_primitive, attrs_primitive = self._forward_primitive(latent_codes_coarse, pts_coarse) # 64, 2048, 1
        dists_primitive = dists_primitive.squeeze(-1)
        
        # calculate loss
        loss_fine = self.lossfun_fine(dists_deepsdf, dists_gt_fine)
        loss_coarse = self.lossfun_coarse(dists_primitive, dists_gt_coarse)
        reg_attr = self._reg_attr(attrs_primitive)
        loss = self._reduce_loss(loss_fine*self.cfg.trainer.loss_fine.weight, loss_coarse*self.cfg.trainer.loss_coarse.weight)
        loss_fine = torch.mean(loss_fine.detach()).item()
        loss_coarse = torch.mean(loss_coarse.detach()).item()
        
        
        (loss + kld*self.cfg.trainer.kld_weight + reg_attr*self.cfg.trainer.attr_reg_weight).backward()
        
        self.optim_deepsdf.step()
        self.optim_primitive.step()
        
        log_info = {'loss': loss.item(), 'loss_fine': loss_fine, 'loss_coarse': loss_coarse, 'reg_attr': reg_attr}
        log_info.update(self.additional_log_info)
        return log_info
    
    def epoch_end(self, epoch, **kwargs):
        self.optim_latentcode.step()
    
    def save(self, epoch, step):
        save_name = "epoch_{}_iters_{}.pth".format(epoch, step)
        path = os.path.join(self.cfg.save_dir, save_name)
        torch.save({
                'trainer_state_dict': self.state_dict(),
                'optim_latentcode_state_dict': self.optim_latentcode.state_dict(),
                'optim_deepsdf_state_dict': self.optim_deepsdf.state_dict(),
                'optim_primitive_state_dict': self.optim_primitive.state_dict(),
                'epoch': epoch,
                'step': step,
            }, path)
    
    def resume(self, ckpt_path):
        print('Resuming {}...'.format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(ckpt['trainer_state_dict'], strict=False)
        # To reduce size, optimizer state dicts are removed from the published check points
        if 'optim_latentcode_state_dict' in ckpt.keys():
            self.optim_latentcode.load_state_dict(ckpt['optim_latentcode_state_dict'])
            self.optim_deepsdf.load_state_dict(ckpt['optim_deepsdf_state_dict'])
            self.optim_primitive.load_state_dict(ckpt['optim_primitive_state_dict'])
        else:
            ckpt['epoch'] = 9999
        return ckpt['epoch']
        
