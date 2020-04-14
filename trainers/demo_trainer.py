import os
import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import itertools

from trainers.base_trainer import BaseTrainer

from trainers.dualsdf_trainer import Trainer as DualSDFTrainer

class Trainer(DualSDFTrainer):
    def _get_render_sdfs(self, zz):
        def sdf_fun(p): # p: N, 3
            z = zz # 1, 256
            N = p.size(0)
            if len(z.shape) == 2:
                z = z.unsqueeze(1)
            z = z.expand(-1,N,-1)
            p = p.unsqueeze(0)
            inp = torch.cat([z, p], dim=-1)
            dists = self.deepsdf_net(inp) # [1 N 1]
            dists = dists.reshape(-1, 1)
            return {'dists':dists}
        
        attrs = self.prim_attr_net(zz)
        def prim_sdf_fun(p):
            p = p.unsqueeze(0)
            if self.prim_sdf_fun.return_idx is True:
                dists, idx = self.prim_sdf_fun(attrs, p)
                dists = dists.reshape(-1, 1)
                return {'dists':dists, 'indices': idx}
            else:
                dists = self.prim_sdf_fun(attrs, p)
                dists = dists.reshape(-1, 1)
                return {'dists':dists}

        return sdf_fun, prim_sdf_fun
        
    def get_known_latent(self, idx):
        num_known_shapes = len(self.sid2idx)
        if idx is None:
            return num_known_shapes
        data_indices = torch.tensor([idx], dtype=torch.long, device=self.device)
        latent_codes_coarse, latent_codes_fine, kld = self._b_idx2latent(self.latent_embeddings, data_indices, num_augment_pts=1) # [64 128]
        loss_kld = torch.mean(0.5 * torch.mean(latent_codes_coarse**2, dim=-1))
        self.stats_loss_kld = loss_kld.item()
        return latent_codes_coarse
    
    def step_manip(self, feature, manip_fun):
        latent_codes_coarse = feature.to(self.device).clone().detach().requires_grad_(True)
        optim, lrscheduler = self._get_optim([latent_codes_coarse], self.cfg.manip.optim)
        
        for i in range(8):
            optim.zero_grad()
            # PrimitiveSDF
            attrs_primitive = self.prim_attr_net(latent_codes_coarse) # 64, 2048, 1
            attrs_primitive = attrs_primitive.reshape(attrs_primitive.size(0), -1, 4) # 64, 256, 4, (r x y z)
            
            loss_manip = manip_fun(attrs_primitive[0], latent_codes_coarse)

            loss_kld = torch.mean(0.5 * torch.mean(latent_codes_coarse**2, dim=-1))
            loss_manip.backward()
            optim.step()
            
        self.stats_loss_kld = loss_kld.item()
        return latent_codes_coarse.detach(), attrs_primitive[0]
        
    def render_express(self, feature):
        latent_codes_fine = feature.to(self.device)
        if not hasattr(self, 'renderer'):
            from toolbox.sdf_renderer import SDFRenderer
            renderer = SDFRenderer(self.cfg.render_web, self.device)
        
        # Prepare DeepSDF for rendering
        self.eval()
        with torch.no_grad():
            sdf_fun, _ = self._get_render_sdfs(latent_codes_fine)
            
            print('R', end='')
            img = renderer.render(sdf_fun, coloridx=None)
            img = img[...,[2,1,0]]
        return img
    
    def resume_demo(self, ckpt_path):
        print('WebDemo Resuming {}...'.format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        self.cfg.train_shape_ids = range(ckpt['trainer_state_dict']['latent_embeddings.weight_mu'].shape[0])
        self.prep_train()
        self.load_state_dict(ckpt['trainer_state_dict'], strict=False)
    
        
    def render_known_shapes(self, **kwargs):
        #writer = kwargs['writer']
        import cv2
        from toolbox.sdf_renderer import SDFRenderer
        print('[RENDER Known SDFs] -- {}'.format(self.cfg.log_dir))
        
        # Prepare DeepSDF for rendering
        self.latent_embeddings.eval()
        self.eval()
        idx2sid = { v:k for k, v in self.sid2idx.items()}
        data_indices = torch.tensor(range(len(self.sid2idx)), dtype=torch.long, device=self.device)
        data_indices_batch = torch.split(data_indices, 1, dim=0)
        with torch.no_grad():
            for data_indices in data_indices_batch:
                latent_codes_coarse, latent_codes_fine, kld = self._b_idx2latent(self.latent_embeddings, data_indices, num_augment_pts=1) # [64 128]
                sdf_fun, prim_sdf_fun = self._get_render_sdfs(latent_codes_fine)
                    
                renderer = SDFRenderer(self.cfg.render, self.device)
                print('R-P', end='')
                img = renderer.render(prim_sdf_fun, coloridx=None)
                #writer.add_image('render_sdfs_prim', img, data_indices.item(), dataformats='HWC')
                img = img[...,[2,1,0]]
                out_path = os.path.join(self.cfg.log_dir, 'render_sdf_{}_prim.png'.format(idx2sid[data_indices.item()]))
                cv2.imwrite(out_path, img,[cv2.IMWRITE_PNG_COMPRESSION,3])
                print('R-D', end='')
                img = renderer.render(sdf_fun, coloridx=None)
                #writer.add_image('render_sdfs_deep', img, data_indices.item(), dataformats='HWC')
                img = img[...,[2,1,0]]
                out_path = os.path.join(self.cfg.log_dir, 'render_sdf_{}.png'.format(idx2sid[data_indices.item()]))
                cv2.imwrite(out_path, img,[cv2.IMWRITE_PNG_COMPRESSION,3])                


