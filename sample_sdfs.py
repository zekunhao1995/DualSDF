import os
import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time

from extensions.mesh2sdf2_cuda import mesh2sdf
from toolbox import pcl_library

def sdfmeshfun(point, mesh):
    out_ker = mesh2sdf.mesh2sdf_gpu(point.contiguous(),mesh)[0]
    return out_ker
    
def trimmesh(mesh_t, residual=False):
    mesh_t = mesh_t.to("cuda:0")
    valid_triangles = mesh2sdf.trimmesh_gpu(mesh_t)
    if residual:
        valid_triangles = ~valid_triangles
    mesh_t = mesh_t[valid_triangles,:,:].contiguous()
    print("[Trimmesh] {} -> {}".format(valid_triangles.size(0),mesh_t.size(0)))
    return mesh_t
    
def meshpreprocess_bsphere(mesh_path):
    mesh = np.load(mesh_path)
    mesh[:,:,1] *= -1
    # normalize mesh
    mesh = mesh.reshape(-1,3)
    mesh_max = np.amax(mesh, axis=0)
    mesh_min = np.amin(mesh, axis=0)
    mesh_center = (mesh_max + mesh_min) / 2
    mesh = mesh - mesh_center
    # Find the max distance to origin
    max_dist = np.sqrt(np.max(np.sum(mesh**2, axis=-1)))
    mesh_scale = 1.0 / max_dist
    mesh *= mesh_scale
    mesh = mesh.reshape(-1,3,3)
    mesh_t = torch.from_numpy(mesh.astype(np.float32)).contiguous()
    return mesh_t

def normalize(x):
    x /= torch.sqrt(torch.sum(x**2))
    return x

def main(args):
    device = torch.device('cuda:0')
    data_path = args.mesh_npy_path

    data_list = []
    with os.scandir(data_path) as npy_list:
        for npy_path in npy_list:
            if npy_path.is_file():
                data_list.append(npy_path.path)
    data_list.sort()
    print(len(data_list))
    num_shapes = len(data_list)

    target_path = args.output_path
    
    # for each mesh, sample points within bounding sphere.
    # According to DeepSDF paper, 250,000x2 points around the surface,
    # 25,000 points within the unit sphere uniformly
    # To sample points around the surface, 
    #   - sample points uniformly on the surface,
    #   - Perturb the points with gaussian noise var=0.0025 and 0.00025
    #   - Then compute SDF
    num_surface_samples = 320000
    num_sphere_samples = 250000
    target_samples = 250000

    noise_vec = torch.empty([num_surface_samples,3], dtype=torch.float32, device=device) # x y z
    noise_vec2 = torch.empty([num_sphere_samples,3], dtype=torch.float32, device=device) # x y z
    noise_vec3 = torch.empty([num_sphere_samples,1], dtype=torch.float32, device=device) # x y z

    for shape_id in range(args.resume, len(data_list)):
        print('Processing {} - '.format(shape_id), end='')
        mesh_path = data_list[shape_id]
        mesh_path_split = mesh_path.split('/')
        classid = mesh_path_split[-2]
        shapeid = mesh_path_split[-1].split('.')[0]
        print(classid, shapeid)
        
        start = time.time()
        
        mesh = meshpreprocess_bsphere(mesh_path).to(device)
        if not args.notrim:
            # Remove inside triangles
            mesh = trimmesh(mesh)
        pcl = torch.from_numpy(pcl_library.mesh2pcl(mesh.cpu().numpy(), num_surface_samples)).to(device) # [N, 3]
        
        # Surface points
        noise_vec.normal_(0, np.sqrt(0.005))
        points1 = pcl + noise_vec
        noise_vec.normal_(0, np.sqrt(0.0005))
        points2 = pcl + noise_vec
        
        # Unit sphere points
        noise_vec2.normal_(0, 1)
        shell_points = noise_vec2 / torch.sqrt(torch.sum(noise_vec2**2, dim=-1, keepdim=True))
        noise_vec3.uniform_(0, 1) # r = 1
        points3 = shell_points * (noise_vec3**(1/3))

        all_points = torch.cat([points1, points2, points3], dim=0)
        
        
        #print(all_points.shape)
        sample_dist = sdfmeshfun(all_points, mesh)
        #print(sample_dist.shape)
        
        xyzd = torch.cat([all_points, sample_dist.unsqueeze(-1)], dim=-1).cpu().numpy()
        
        xyzd_sur = xyzd[:num_surface_samples*2]
        xyzd_sph = xyzd[num_surface_samples*2:]
        
        inside_mask = (xyzd_sur[:,3] <= 0)
        outside_mask = np.logical_not(inside_mask)

        inside_cnt = np.count_nonzero(inside_mask)
        outside_cnt = np.count_nonzero(outside_mask)
        inside_stor = [xyzd_sur[inside_mask,:]]
        outside_stor = [xyzd_sur[outside_mask,:]]
        n_attempts = 0
        badsample = False
        while (inside_cnt < target_samples) or (outside_cnt < target_samples):
            noise_vec.normal_(0, np.sqrt(0.005))
            points1 = pcl + noise_vec
            noise_vec.normal_(0, np.sqrt(0.0005))
            points2 = pcl + noise_vec
            all_points = torch.cat([points1, points2], dim=0)
            sample_dist = sdfmeshfun(all_points, mesh)
            xyzd_sur = torch.cat([all_points, sample_dist.unsqueeze(-1)], dim=-1).cpu().numpy()
            inside_mask = (xyzd_sur[:,3] <= 0)
            outside_mask = np.logical_not(inside_mask)
            inside_cnt += np.count_nonzero(inside_mask)
            outside_cnt += np.count_nonzero(outside_mask)
            inside_stor.append(xyzd_sur[inside_mask,:])
            outside_stor.append(xyzd_sur[outside_mask,:])
            n_attempts += 1
            print(" - {}nd Attempt: {} / {}".format(n_attempts, inside_cnt, target_samples))
            if n_attempts > 200 or ((np.minimum(inside_cnt, outside_cnt)/n_attempts) < 500):
                with open('bads_list_{}.txt'.format(classid), 'a+') as f:
                    f.write('{},{},{},{}\n'.format(classid, shapeid, np.minimum(inside_cnt, outside_cnt), n_attempts))
                badsample = True
                break
            
        xyzd_inside = np.concatenate(inside_stor, axis=0)
        xyzd_outside = np.concatenate(outside_stor, axis=0)
        
        num_yields = np.minimum(xyzd_inside.shape[0], xyzd_outside.shape[0])
        xyzd_inside = xyzd_inside[:num_yields,:]
        xyzd_outside = xyzd_outside[:num_yields,:]
        
        xyzd = np.concatenate([xyzd_inside, xyzd_outside], axis=0)
        
        end = time.time()
        print("[Perf] time: {}, yield: {}".format(end - start, num_yields))
        
        save_path = os.path.join(target_path, classid+"_surface")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path,'{}.npy'.format(shapeid)), xyzd)

        save_path = os.path.join(target_path, classid+"_sphere")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path,'{}.npy'.format(shapeid)), xyzd_sph)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sample SDF values from meshes. All the NPY files under mesh_npy_path and its child dirs will be converted and the directory structure will be preserved.')
    parser.add_argument('mesh_npy_path', type=str,
                        help='The dir containing meshes in NPY format [ #triangles x 3(vertices) x 3(xyz) ]')
    parser.add_argument('output_path', type=str,
                        help='The output dir containing sampled SDF in NPY format [ #points x 4(xyzd) ]')
    parser.add_argument('--notrim', default=False, action='store_true')
    parser.add_argument('--resume', type=int, default=0)
    args = parser.parse_args()
    main(args)
    
    
    
    
