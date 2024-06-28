<div align='center'>

<h2><a href="https://ieeexplore.ieee.org/abstract/document/10462911">Image Understands Point Cloud: Weakly Supervised 3D Semantic Segmentation via Association Learning (TIP2024)</a></h2>

Tianfang Sun<sup>1</sup>, Zhizhong Zhang<sup>1</sup>, Xin Tan<sup>1</sup>, Yanyun Qu<sup>2</sup>, Yuan Xie<sup>1</sup>
<br>
 
<sup>1</sup>ECNU, <sup>2</sup>XMU
 
</div>

# Installation

For easy installation, we recommend using [conda](https://www.anaconda.com/):

```shell
conda create -n iupc python=3.9
conda activate iupc
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip3 install numba nuscenes-devkit
# We need opencv-contrib-python to perform superpixel segment and it is not compatible with opencv-python
pip3 uninstall opencv-python
pip3 install opencv-contrib-python
 # We need open3d and hdbscan to perform point cloud pre-segmentation
pip3 install open3d
pip3 install hdbscan
pip3 install tensorboard
```

Our method is based on [torchpack](https://github.com/zhijian-liu/torchpack) and [torchsparse](https://github.com/mit-han-lab/torchsparse). To install torchpack, we recommend to firstly install openmpi and mpi4py.

```shell
conda install -c conda-forge mpi4py openmpi
```

Install torchpack

```shell
pip install git+https://github.com/zhijian-liu/torchpack.git
```

Before installing torchsparse, install [Google Sparse Hash](https://github.com/sparsehash/sparsehash) library first.

```shell
sudo apt install libsparsehash-dev
```

Then install torchsparse (v1.4.0) by

```shell
pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

# Model Preparation

Please download ImageNet pretrained weight for SwiftNet from [Google Drive](https://drive.google.com/file/d/17Z5fZMcaSkpDdcm6jDBaXBFSo1eDxAsD/view?usp=sharing) or [BaiduDisk](https://pan.baidu.com/s/17Wn_zj69v1_QdjAP1v7eMw?pwd=063m).

# Data Preparation

Please download the datasets following the official instruction. The official websites of each dataset are listed as following: [nuScenes_lidarseg](https://www.nuscenes.org/nuscenes#download), [SemanticKITTI](http://www.semantic-kitti.org/dataset.html), [Waymo open](https://waymo.com/open/).
The color images of SemanticKITTI datasets can be downloaded from [KITTI-odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset.

## Weak Label Preparation

### nuScenes_lidarseg

Please run the following command to cluster and generate weak labels for the nuScenes_lidarseg dataset. The whole process will take several hours 
depended on the CPU.

```bash
python3 prepare_sparse_label.py configs/nuscenes/pretrain/preseg.yaml
```

### Waymo

### SemanticKITTI

### Result

|    Statistics    | nuScenes | Waymo | SemanticKITTI |
|:----------------:|:--------:|:-----:|:-------------:|
|   Sparse label   |   0.8%   | 0.3%  |     0.08%     |
| Propagated label |   48.5%  | 21.2% |     22.7%     |
|  Negative label  |   44.5%  | 76.7% |     70.6%     |

## Superpixel Preparation

### nuScenes_lidarseg

Please run the following command to generate superpixels for the nuScenes_lidarseg dataset. The whole process will take several hours 
depended on the CPU.

```bash
python3 prepare_nusc_superpixel.py
```

# Training

## nuScenes_lidarseg

1. Run the following command to train the model.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchpack dist-run -np 3 python3 train_lc_sparse_assop2p_nusc.py configs/nuscenes/pretrain/assop2p.yaml --run-dir runs/iupc_nusc/assop2pw0p5_visw0p5_sybn
```
2. For the first round, run the following command to generate pseudo label.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchpack dist-run -np 3 python3 train_lc_estep_nusc.py configs/nuscenes/estep/it1.yaml --run-dir runs/iupc_nusc/assop2pw0p5_visw0p5_sybn --weight-path runs/iupc_nusc/assop2pw0p5_visw0p5_sybn/checkpoints/max-iou-vox-val.pt
```

3. Run the following command to use the generated pseudo label for training.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchpack dist-run -np 3 python3 train_lc_sparse_assop2p_mstep_nusc.py configs/nuscenes/mstep/assop2p.yaml --run-dir runs/iupc_nusc/assop2pw0p5_visw0p5_sybn
```

4. For the second round, run the following command to generate pseudo label.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchpack dist-run -np 3 python3 train_lc_estep_nusc_2.py configs/nuscenes/estep/it2.yaml --run-dir runs/iupc_nusc/assop2pw0p5_visw0p5_sybn --weight-path runs/iupc_nusc/assop2pw0p5_visw0p5_sybn/checkpoints_mstep_0/max-iou-vox-val.pt
```

5. Run the following command to use the generated pseudo label for training.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchpack dist-run -np 3 python3 train_lc_sparse_assop2p_mstep_nusc.py configs/nuscenes/mstep/assop2p_it2.yaml --run-dir runs/iupc_nusc/assop2pw0p5_visw0p5_sybn
```

# Results

## nuScenes_lidarseg

|               | Iter 0 | Iter 1 | Iter 2 | Full |
|:-------------:|:------:|:------:|:------:|:----:|
|  mIoU(paper)  |  77.1  |  77.3  |  77.5  | 78.8 |
| mIoU(reprod.) |  76.7  |  76.9  |  77.4  | 78.6 |

# TODOs

 - [x] Test and release the codes for nuScenes datasets.
 - [ ] Test and release the codes for SemanticKITTI datasets.
 - [ ] Test and release the codes for Waymo datasets.