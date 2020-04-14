# DualSDF

This repo contains an implimentation of the CVPR 2020 paper:  

**DualSDF: Semantic Shape Manipulation using a Two-Level Representation**,  
by **Zekun Hao**, **Hadar Averbuch-Elor**, **Noah Snavely**, **Serge Belongie**.  

[[Paper]](https://arxiv.org/pdf/2004.02869.pdf) 
[[Website]](http://www.cs.cornell.edu/~hadarelor/dualsdf/)

## Live demo
[[Demo-Chair]](http://35.202.137.6:8080/) [[Demo-Airplane]](http://35.202.137.6:8081/)

[![DualSDF Demo](dualsdf_demo_snap.png)](http://35.202.137.6:8080/)  

## Running the live demo locally
We provide pretrained models for chair and airplane categories. Following [Park et al.](https://github.com/facebookresearch/DeepSDF), each model is trained on a subset of shapes from the corresponding [ShapeNet](https://www.shapenet.org/) category.

```bash
# Host the chair demo on port 1234 (default port) using GPU 0
python3.6 demo.py ./config/dualsdf_chairs_demo.yaml --pretrained ./pretrained/dualsdf_chairs_demo/epoch_2799.pth

# Host the airplane demo on port 4321 using GPU 1
CUDA_VISIBLE_DEVICES=1 python3.6 demo.py ./config/dualsdf_airplanes_demo.yaml --pretrained ./pretrained/dualsdf_airplanes_demo/epoch_2799.pth --port 4321

# Visit localhost:1234 or localhost:4321 on a browser that supports WebGL.
# We have tested Firefox and Chrome.
```

## Training DualSDF from scratch
### Data preparation

#### Download ShapeNetCore v2 dataset

[ShapeNet website](https://www.shapenet.org/)

#### Convert ShapeNet meshes to numpy npy files

For ease of process, we convert all the meshes to numpy arrays before sampling SDFs.

[Tools](https://www.shapenet.org/tools) are widely available to load ShapeNet obj files. Each npy file should contain a float32 array with a shape of `#triangles x 3 (vertices) x 3 (xyz coordinate of each vertex)`, representing a triangle mesh.

The naming convention is as follows:
```
<category id>/
    <shape_id>.npy
```

#### Sample signed distance fields from meshes

Compile the CUDA kernel for computing SDF:  
```bash
cd extensions/mesh2sdf2_cuda
make
```

Sample SDFs using the provided script:
```bash
python3.6 sample_sdfs.py <path_to_mesh_npy_folder> <path_to_save_sampled_results>
```
The results should look like this:
```
<path_to_save_sampled_results>/
    <shape_id>_sphere/
        <shape_id>.npy
    <shape_id>_surface/
        <shape_id>.npy
```

### Training
#### Creating a config file
It is a good idea to start with an existing config file (i.e. `config/dualsdf_airplanes_demo.yaml`). Edit the `data` section to reflex your dataset configuration. You will need to make new split files for new datasets.
```
data:
    ...
    cate_id: "02691156"
    split_files:
        train: ./datasets/splits/sv2_planes_all.json
        test: ./datasets/splits/sv2_planes_all.json
    sdf_data_dir:
        surface: /mnt/data3/ShapeNet_sdf_even/02691156_surface
        sphere: /mnt/data3/ShapeNet_sdf_even/02691156_sphere
    ...
...
```
#### Start training
```bash
CUDA_VISIBLE_DEVICES=2 python3.6 train.py ./config/dualsdf_airplanes_demo.yaml
```
Tensorboard databases and checkpoints will appear under `logs` directory.

#### Rendering SDFs
To render shape reconstruction results on the training set, run the following command with properly set paths to config file and checkpoint file:
```bash
CUDA_VISIBLE_DEVICES=2 python3.6 train.py ./config/dualsdf_airplanes_demo.yaml --resume --pretrained ./pretrained/dualsdf_airplanes_demo/epoch_2799.pth --special render_known_shapes
```
The rendered images of both primitive-based representation and high-resolution representation will appear under the `logs/special_render_known_shapes_dualsdf_airplanes_demo_<datetime>` directory. Many options related to rendering can be modified in the config file.


## Citing DualSDF

If you find our code useful, please consider citing our paper:

```
@article{zekun2020dualsdf,
 title={DualSDF: Semantic Shape Manipulation using a Two-Level Representation},
 author={Hao, Zekun and Averbuch-Elor, Hadar and Snavely, Noah and Belongie, Serge},
 journal={arXiv},
 year={2020}
}
```
