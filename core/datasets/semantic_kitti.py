import os
import os.path

import yaml
import numpy as np
from PIL import Image

from torchvision.transforms import transforms

import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn, sparse_collate
from copy import deepcopy

__all__ = ['SemanticKITTI']

label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


class SemanticKITTI(dict):

    def __init__(self, root, voxel_size, **kwargs):
        config_path = os.path.join(root, 'semantic-kitti.yaml')
        with open(config_path, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        learning_map = semkittiyaml['learning_map']
        root = os.path.join(root, 'sequences')
        super().__init__({
            'train': _SemanticKITTIInternal(root, voxel_size, split='train', learning_map=learning_map),
            'val': _SemanticKITTIInternal(root, voxel_size, split='val', learning_map=learning_map)
        })


class _SemanticKITTIInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 split,
                 learning_map):
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.labels_mapping = learning_map
        self.seqs = []
        if split == 'train':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

        self.pcd_files = []
        for seq in self.seqs:
            for pcd_name in sorted(os.listdir(os.path.join(self.root, seq, 'velodyne'))):
                self.pcd_files.append(os.path.join(self.root, seq, 'velodyne', str(pcd_name)))

    def __len__(self):
        return len(self.pcd_files)

    def _load_pcd(self, index):
        filepath = self.pcd_files[index]
        pts = np.fromfile(filepath, dtype=np.float32).reshape((-1, 4))
        # img = Image.open(self.img_files[index]).convert('RGB')
        if self.split == 'test':
            labels_ = np.expand_dims(np.zeros_like(pts[:, 0], dtype=int), axis=1)
        else:
            lidar_label_path = filepath.replace('velodyne', 'labels')[:-3] + 'label'
            labels_ = np.fromfile(lidar_label_path, dtype=np.int32).reshape([-1, 1])
            labels_ = labels_ & 0xFFFF
            labels_ = np.vectorize(self.labels_mapping.__getitem__)(labels_).flatten()
        return pts, labels_
        # return pts, labels_, img

    def __getitem__(self, index):
        pts, labels_ = self._load_pcd(index)

        # if 'train' in self.split:
        #     flip_type = np.random.choice(4, 1)
        #     if flip_type == 1:
        #         pts[:, 0] = -pts[:, 0]
        #     elif flip_type == 2:
        #         pts[:, 1] = -pts[:, 1]
        #     elif flip_type == 3:
        #         pts[:, :2] = -pts[:, :2]

        pts_cp = np.zeros_like(pts)
        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
            pts_cp[:, :3] = np.dot(pts[:, :3], rot_mat) * scale_factor
        else:
            theta = 0.0
            transform_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                      [-np.sin(theta), np.cos(theta), 0],
                                      [0, 0, 1]])
            pts_cp[...] = pts[...]
            pts_cp[:, :3] = np.dot(pts_cp[:, :3], transform_mat)

        pts_cp[:, 3] = pts[:, 3]  # block为经过随机旋转和放缩的点云 [N, 4] -> x,y,z,sig
        voxel = np.round(pts_cp[:, :3] / self.voxel_size).astype(np.int32)  # voxelization
        voxel -= voxel.min(0, keepdims=1)  # 将体素坐标最小值为0

        feat_ = pts_cp

        _, inds, inverse_map = sparse_quantize(voxel,
                                               return_index=True,
                                               return_inverse=True)

        voxel_full = voxel[inds]
        feat_full = feat_[inds]
        labels_full = labels_[inds]
        lidar = SparseTensor(feat_full, voxel_full)
        labels = SparseTensor(labels_full, voxel_full)
        labels_ = SparseTensor(labels_, voxel)
        inverse_map = SparseTensor(inverse_map, voxel)

        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
        }

    @staticmethod
    def collate_fn(batch):
        ans_dict = {}
        for key in batch[0].keys():
            if key == "masks":
                ans_dict[key] = [torch.from_numpy(sample[key]) for sample in batch]
            elif key == "pixel_coordinates":
                ans_dict[key] = [torch.from_numpy(sample[key]).float() for sample in batch]
            elif isinstance(batch[0][key], SparseTensor):
                ans_dict[key] = sparse_collate(
                    [sample[key] for sample in batch])  # sparse_collate_tensor -> sparse_collate
            elif isinstance(batch[0][key], np.ndarray):
                ans_dict[key] = torch.stack(
                    [torch.from_numpy(sample[key]).float() for sample in batch], dim=0)
            elif isinstance(batch[0][key], torch.Tensor):
                ans_dict[key] = torch.stack([sample[key] for sample in batch], dim=0)
            elif isinstance(batch[0][key], dict):
                ans_dict[key] = sparse_collate_fn(
                    [sample[key] for sample in batch])
            else:
                ans_dict[key] = [sample[key] for sample in batch]
        return ans_dict
