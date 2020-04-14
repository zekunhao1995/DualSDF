# DualSDF

This repo contains an implimentation of the CVPR 2020 paper:  

**DualSDF: Semantic Shape Manipulation using a Two-Level Representation**,  
by **Zekun Hao**, **Hadar Averbuch-Elor**, **Noah Snavely**, **Serge Belongie**.  

[[Paper]](https://arxiv.org/pdf/2004.02869.pdf) 
[[Website]](http://www.cs.cornell.edu/~hadarelor/dualsdf/)

## Live demo
[[Demo-Chair]](http://35.202.137.6:8080/) [[Demo-Airplane]](http://35.202.137.6:8081/)

<p float="left">
    <img src="dualsdf_demo_snap.png"/>
</p>

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
