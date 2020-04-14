import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFRenderer():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.setup_camera()
        
    # ray_dir: N H W 3
    @staticmethod
    def cam_lookat(ray_dir, ray_ori, rot_x, rot_y):
        sin_a = np.sin(rot_x)
        cos_a = np.cos(rot_x)
        sin_b = np.sin(rot_y)
        cos_b = np.cos(rot_y)
        # First rotate y, then rotate x
        rot_mat = torch.tensor([cos_b, 0, -sin_b, sin_a*sin_b, cos_a, sin_a*cos_b, cos_a*sin_b, -sin_a, cos_a*cos_b], dtype=torch.float32, device=ray_dir.device)
        # without transpose, first rotate x, then rotate y.
        rot_mat = rot_mat.reshape(3,3) # .transpose(1,0)
        
        ray_dir = torch.matmul(ray_dir, rot_mat)
        ray_ori = torch.matmul(ray_ori, rot_mat)
        
        return ray_dir, ray_ori
    
    # Initialize raytravel to start from the bounding sphere surface.
    # ray_dir: [1,target_res[1],target_res[0],3]
    def init_rt_sph(self, ray_dir, ray_ori, r=1.0):
        a = torch.sum(ray_dir*ray_dir, dim=-1)
        b = 2*torch.sum((ray_dir*ray_ori), dim=-1)
        c = torch.sum(ray_ori*ray_ori, dim=-1)-r**2
        delta = b**2 - 4*a*c
        sol1 = (-b - torch.sqrt(delta)) / (2*a) # -: near sphere surface. +: far sphere surface
        sol2 = (-b + torch.sqrt(delta)) / (2*a) # -: near sphere surface. +: far sphere surface
        ray_travel = sol1.unsqueeze(-1)
        ray_travel_far = sol2.unsqueeze(-1)
        return ray_travel, ray_travel_far
        
    # input: [N H W 3] wavefront position
    # output: [N H W 1] distances
    # output: [N H W 1] primitive index
    def scene_fun(self, sdf_fun, ray_pos):
        net_input = ray_pos.reshape(-1, 3)
        net_output = sdf_fun(net_input)
        dists = net_output['dists']
        if 'indices' in net_output.keys():
            idx = net_output['indices']
            idx = idx.reshape(ray_pos.size(0),ray_pos.size(1),ray_pos.size(2),1)
        else:
            idx = None
        march_dist = dists.reshape(ray_pos.size(0),ray_pos.size(1),ray_pos.size(2),1)
        return march_dist, idx # [1, 512, 768, 1]

    def setup_camera(self):
        target_res = self.cfg.resolution # w h
        ver_scale = self.cfg.ver_scale
        hor_scale = ver_scale / target_res[1] * target_res[0]
        
        # Step 1: get rays
        # [N H W xyz]
        if self.cfg.cam_model.lower() == 'orthographic':
            # Orthographic camera
            ray_dir = torch.zeros([1,target_res[1],target_res[0],3], device=self.device) # x y z
            ray_dir[:,:,:,2] = 1.0
            ray_ori = torch.empty([1,target_res[1],target_res[0],3], device=self.device) # x y z
            ray_ori[:,:,:,0] = torch.linspace(-(hor_scale/2.0),(hor_scale/2.0),target_res[0]).unsqueeze(0).unsqueeze(0)
            ray_ori[:,:,:,1] = torch.linspace(-(ver_scale/2.0),(ver_scale/2.0),target_res[1]).unsqueeze(0).unsqueeze(-1)
            ray_ori[:,:,:,2] = -1.0
        elif self.cfg.cam_model.lower() == 'perspective':
            # Perspective camera
            focal_len = 1.7
            ray_dir = torch.empty([1,target_res[1],target_res[0],3], device=self.device) # x y z
            ray_dir[:,:,:,0] = torch.linspace(-(hor_scale/2.0),(hor_scale/2.0),target_res[0]).unsqueeze(0).unsqueeze(0)
            ray_dir[:,:,:,1] = torch.linspace(-(ver_scale/2.0),(ver_scale/2.0),target_res[1]).unsqueeze(0).unsqueeze(-1)
            ray_dir[:,:,:,2] = focal_len
            ray_dir = F.normalize(ray_dir, dim=-1)
            ray_ori = torch.zeros([1,target_res[1],target_res[0],3], device=self.device) # x y z
            ray_ori[:,:,:,2] = -2.0
        else:
            raise NotImplementedError("Camera model not recognised: {}".format(self.cfg.cam_model))
        
        # Rotate camera
        ray_dir_w, ray_ori_w = self.cam_lookat(ray_dir, ray_ori, self.cfg.rot_ver_deg*np.pi/180, self.cfg.rot_hor_deg*np.pi/180) # ray_dir, ray_ori, rot_x, rot_y
        # Init ray travel
        ray_travel, ray_travel_far = self.init_rt_sph(ray_dir_w, ray_ori_w, r=self.cfg.bsphere_r)
        bg_mask = torch.isnan(ray_travel) # [1,target_res[1],target_res[0],1], bg == True
        self.ray_dir_w = ray_dir_w
        self.ray_ori_w = ray_ori_w
        self.ray_travel = ray_travel
        self.ray_travel_far = ray_travel_far
        self.bg_mask = bg_mask
        
    # cfg:
    # cam_model: Orthographic # Perspective
    # ver_scale: 2.0
    # rot_hor_deg: 60
    # rot_ver_deg: -28.647889
    # sdf_iso_level: 0.004
    # sdf_clamp: 0.1
    # sdf_gain: 1.0
    # colorcoord: 256, [r]xyzrgb, for coloring DeepSDF (radius optional).
    # coloridx: 256, rgb, for coloring spheres and capsules
    # Set them to None for default shading (blue sky, red sun)
    def render(self, sdf_fun, coloridx=None, colorcoord=None):
        with torch.no_grad():
            ray_travel = self.ray_travel
            ray_travel_far = self.ray_travel_far
            bg_mask = self.bg_mask
            ray_dir_w = self.ray_dir_w
            ray_ori_w = self.ray_ori_w
            
            # Ray marching
            for march_step in range(self.cfg.steps):
                print('.',end='', flush=True)
                ray_pos = ray_ori_w + ray_travel * ray_dir_w
                march_dist, idx = self.scene_fun(sdf_fun, ray_pos)
                march_dist -= self.cfg.sdf_iso_level
                march_dist = torch.clamp(march_dist, -self.cfg.sdf_clamp, self.cfg.sdf_clamp)
                ray_travel += march_dist*self.cfg.sdf_gain
                ray_travel = torch.min(ray_travel, ray_travel_far)
            print('*')
            bg_mask = bg_mask | ((ray_travel_far - ray_travel) < 1e-5)
            
            ray_pos = ray_ori_w + ray_travel * ray_dir_w
            if self.cfg.numerical_normal:
                # https://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
                eps = 5e-4
                k1 = torch.tensor([1,-1,-1], dtype=torch.float32, device=self.device)
                k2 = torch.tensor([-1,-1,1], dtype=torch.float32, device=self.device)
                k3 = torch.tensor([-1,1,-1], dtype=torch.float32, device=self.device)
                k4 = torch.tensor([1,1,1], dtype=torch.float32, device=self.device)
                normals = F.normalize( \
                    k1*self.scene_fun(sdf_fun, ray_pos + eps*k1)[0] + \
                    k2*self.scene_fun(sdf_fun, ray_pos + eps*k2)[0] + \
                    k3*self.scene_fun(sdf_fun, ray_pos + eps*k3)[0] + \
                    k4*self.scene_fun(sdf_fun, ray_pos + eps*k4)[0], dim=-1)
            else:                        
                # Autograd surface normal
                with torch.enable_grad():
                    ray_pos.requires_grad=True
                    dd = self.scene_fun(sdf_fun, ray_pos)[0]
                    dd.backward(torch.ones_like(dd) + dd*0.0, retain_graph=True)
                    normals = F.normalize(ray_pos.grad, dim=-1)
                    ray_pos.requires_grad=False
            
            # Shading
            # Assuming lambertian surface
            sun_dir = F.normalize(torch.tensor([0.8,-0.4,0.2], dtype=torch.float32, device=self.device), dim=0)
            sun_color = torch.tensor([1.0,0.7,0.5], dtype=torch.float32, device=self.device)
            sun_dif = torch.clamp(torch.matmul(normals, sun_dir).unsqueeze(-1), 0.0, 1.0)
            
            sky_dir = torch.tensor([0.0,-1.0,0.0], dtype=torch.float32, device=self.device)
            sky_color = torch.tensor([0.0,0.1,0.3], dtype=torch.float32, device=self.device)
            sky_dif = torch.clamp(0.5+0.5*torch.matmul(normals, sky_dir).unsqueeze(-1), 0.0, 1.0)
            
            # color the primitives
            # coloridx: [#prims, rgb]
            # idx: [1 h w 1]
            if coloridx is not None:
                colors = coloridx[idx.squeeze(-1)]
                colors1 = colors ** 2.2  * 0.5
                colors2 = colors1
            # color the mesh
            elif colorcoord is not None: # 256, xyzrgb
                raise NotImplementedError("[SDFRenderer] colorcoord is broken. Do not use.")
            elif self.cfg.fg_color is not None:
                colors1 = torch.tensor(self.cfg.fg_color, dtype=torch.float32, device=self.device) ** 2.2  * 0.5
                colors2 = colors1
            else:
                colors1 = sun_color
                colors2 = sky_color
    
            framebuffer = sun_dif * colors1
            framebuffer += sky_dif * colors2
            framebuffer = (framebuffer)**(1/2.2)
            

            framebuffer[bg_mask[...,0], :] = torch.tensor(self.cfg.bg_color, dtype=torch.float32, device=self.device)
            img = (torch.clamp(framebuffer,0,1)[0].cpu().numpy()*255).astype(np.uint8)

        return img

    
