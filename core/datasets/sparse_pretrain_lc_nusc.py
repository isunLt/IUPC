import os
import numpy as np

import torch
from torch.utils import data

from nuscenes import NuScenes as NuScenes_devkit
from torchvision.transforms import transforms

from torchpack.environ import get_run_dir
from torchpack.utils.config import configs

from core.datasets.utils import PCDTransformTool, GaussianBlur, fetch_color
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn, sparse_collate
from PIL import Image
from pyquaternion import Quaternion

CLASS_WEIGHT = [
    0,
    10.6310,
    29.8912,
    14.3095,
    5.4934,
    20.9552,
    27.1116,
    18.4044,
    24.8379,
    13.9495,
    8.3447,
    1.9305,
    11.0304,
    4.0755,
    4.0729,
    2.5711,
    3.0951
]

NUSCENES_MAP_LABEL2NAME_16 = {
    0: 'noise',
    1: 'barrier',
    2: 'bicycle',
    3: 'bus',
    4: 'car',
    5: 'construction_vehicle',
    6: 'motorcycle',
    7: 'pedestrian',
    8: 'traffic_cone',
    9: 'trailer',
    10: 'truck',
    11: 'driveable_surface',
    12: 'other_flat',
    13: 'sidewalk',
    14: 'terrain',
    15: 'manmade',
    16: 'vegetation',
}


class NuScenesLCSparseLabel(dict):
    def __init__(self, root, voxel_size, version, verbose, **kwargs):
        self.nusc = NuScenes_devkit(dataroot=root, version=version, verbose=verbose)
        im_cr = kwargs.get("image_crop_rate", 0.5)
        pts_sr = kwargs.get('pts_sr', 1.0)
        use_color = kwargs.get('use_color', False)
        use_augment = kwargs.get('use_augment', False)
        # superpixel = kwargs.get('superpixel', None)
        # sample_rate = kwargs.get('sample_rate', None)
        super(NuScenesLCSparseLabel, self).__init__({
            "train": _NuScenesInternalLCSparseLabel(nusc=self.nusc, voxel_size=voxel_size, split="train",
                                                    image_crop_rate=im_cr, pts_sr=pts_sr,
                                                    use_color=use_color,
                                                    use_augment=use_augment),
            "val": _NuScenesInternalLCSparseLabel(nusc=self.nusc, voxel_size=voxel_size, split="val",
                                                  image_crop_rate=im_cr, pts_sr=1.0, use_color=use_color,
                                                  use_augment=False)
        })


class _NuScenesInternalLCSparseLabel(data.Dataset):
    labels_mapping = {
        1: 0,
        5: 0,
        7: 0,
        8: 0,
        10: 0,
        11: 0,
        13: 0,
        19: 0,
        20: 0,
        0: 0,
        29: 0,
        31: 0,
        9: 1,
        14: 2,
        15: 3,
        16: 3,
        17: 4,
        18: 5,
        21: 6,
        2: 7,
        3: 7,
        4: 7,
        6: 7,
        12: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        30: 16
    }

    CAM_CHANNELS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    IMAGE_SIZE = (900, 1600)

    CLASS_DISTRIBUTE = [2.87599770e-01, 7.84806427e-03, 1.19217528e-04, 3.88372281e-03, 3.21376629e-02, 1.27727921e-03,
                        3.60467902e-04, 1.95227505e-03, 6.20954881e-04, 4.13906749e-03, 1.33608580e-02, 2.67327832e-01,
                        7.21896959e-03, 5.92055787e-02, 5.92833998e-02, 1.50278018e-01, 1.03386862e-01]

    def __init__(self, nusc, voxel_size, split="train", **kwargs):
        self.nusc = nusc
        self.voxel_size = voxel_size
        self.split = split
        self.ignored_labels = np.sort([0])
        im_cr = configs['dataset']['image_crop_rate']
        self.input_image_size = [int(x * im_cr) for x in self.IMAGE_SIZE]
        self.transform = transforms.Compose([transforms.Resize(size=self.input_image_size)])
        self.augment = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])
        if self.split == 'train':
            # self.valid_cam_channel = ['CAM_FRONT']
            self.valid_cam_channel = self.CAM_CHANNELS
        else:
            self.valid_cam_channel = self.CAM_CHANNELS

        self.use_color = kwargs.get('use_color', False)
        self.use_augment = kwargs.get('use_augment', False)
        self.run_dir = get_run_dir()
        self.num_classes = 17
        self.weak_label_dir = configs['dataset']['sparse_label_path']
        # self.weak_label_dir = './data/nuscenes/pano_labels'
        self.sparse_label_dir = os.path.join(self.weak_label_dir, 'sparse_label')
        self.prop_label_dir = os.path.join(self.weak_label_dir, 'prop_label')
        self.neg_label_dir = os.path.join(self.weak_label_dir, 'neg_label')
        assert os.path.exists(self.sparse_label_dir)
        assert os.path.exists(self.prop_label_dir)
        assert os.path.exists(self.neg_label_dir)

        if self.split == "train":
            select_idx = np.load("./data/nuscenes/nuscenes_train_official.npy")
            self.sample = [self.nusc.sample[i] for i in select_idx]
        elif self.split == "val":
            select_idx = np.load("./data/nuscenes/nuscenes_val_official.npy")
            self.sample = [self.nusc.sample[i] for i in select_idx]
        elif self.split == "test":
            self.sample = self.nusc.sample
        else:
            print("split not implement yet, exit!")
            exit(-1)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sample)
        # return 20

    def __getitem__(self, index):
        sample = self.sample[index]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_channel = self.nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_channel["filename"])
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])[:, :4]  # N, 4
        camera_channel = []
        pixel_coordinates = []  # 6, N, 2
        masks = []
        valid_mask = np.array([-1] * pts.shape[0])

        for idx, channel in enumerate(self.valid_cam_channel):
            cam_token = sample['data'][channel]
            cam_channel = self.nusc.get('sample_data', cam_token)
            im = Image.open(os.path.join(self.nusc.dataroot, cam_channel['filename'])).convert('RGB')
            if self.use_augment:
                camera_channel.append(np.array(self.augment(self.transform(im))))
            else:
                camera_channel.append(np.array(self.transform(im)))
            pcd_trans_tool = PCDTransformTool(pts[:, :3])
            # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
            cs_record = self.nusc.get('calibrated_sensor', lidar_channel['calibrated_sensor_token'])
            pcd_trans_tool.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            pcd_trans_tool.translate(np.array(cs_record['translation']))
            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get('ego_pose', lidar_channel['ego_pose_token'])
            pcd_trans_tool.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
            pcd_trans_tool.translate(np.array(poserecord['translation']))
            # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
            poserecord = self.nusc.get('ego_pose', cam_channel['ego_pose_token'])
            pcd_trans_tool.translate(-np.array(poserecord['translation']))
            pcd_trans_tool.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get('calibrated_sensor', cam_channel['calibrated_sensor_token'])
            pcd_trans_tool.translate(-np.array(cs_record['translation']))
            pcd_trans_tool.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
            mask = np.ones(pts.shape[0], dtype=bool)
            mask = np.logical_and(mask, pcd_trans_tool.pcd[2, :] > 1)
            # Fifth step: project from 3d coordinate to 2d coordinate
            pcd_trans_tool.pcd2image(np.array(cs_record['camera_intrinsic']))
            pixel_coord = pcd_trans_tool.pcd[:2, :]
            pixel_coord[0, :] = pixel_coord[0, :] / (im.size[0] - 1.0) * 2.0 - 1.0  # width
            pixel_coord[1, :] = pixel_coord[1, :] / (im.size[1] - 1.0) * 2.0 - 1.0  # height
            # pixel_coordinates.append(pixel_coord.T)

            # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
            # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            mask = np.logical_and(mask, pixel_coord[0, :] > -1)
            mask = np.logical_and(mask, pixel_coord[0, :] < 1)
            mask = np.logical_and(mask, pixel_coord[1, :] > -1)
            mask = np.logical_and(mask, pixel_coord[1, :] < 1)
            valid_mask[mask] = idx
            masks.append(mask)
            pixel_coordinates.append(pixel_coord.T)

        pt_with_img_idx = (valid_mask != -1)
        pts = pts[pt_with_img_idx]
        pixel_coordinates = np.stack([coord[pt_with_img_idx] for coord in pixel_coordinates], axis=0)
        masks = np.stack([m[pt_with_img_idx] for m in masks], axis=0)
        camera_channel = np.stack(camera_channel, axis=0)

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
        if self.split == "test":
            labels_ = np.expand_dims(np.zeros_like(pts[:, 0], dtype=int), axis=1)
        else:
            lidar_label_path = os.path.join(self.nusc.dataroot, self.nusc.get("lidarseg", lidar_token)["filename"])
            label_raw_ = np.fromfile(lidar_label_path, dtype=np.uint8)
            labels_ = label_raw_[pt_with_img_idx].reshape([-1, 1])
            labels_ = np.vectorize(self.labels_mapping.__getitem__)(labels_).flatten()
            if self.split == 'train':
                sparse_label_path = os.path.join(self.sparse_label_dir, str(lidar_token) + '_sparse_label.bin')
                sparse_label_mask = np.fromfile(sparse_label_path, dtype=np.uint8)[pt_with_img_idx].astype(bool)
                sparse_label = np.full_like(labels_, fill_value=self.ignored_labels[0], dtype=np.uint8)
                sparse_label[sparse_label_mask] = labels_[sparse_label_mask]
                prop_label_path = os.path.join(self.prop_label_dir, str(lidar_token) + '_prop_label.bin')
                prop_label = np.fromfile(prop_label_path, dtype=np.uint8)[pt_with_img_idx]
                prop_label[sparse_label_mask] = sparse_label[sparse_label_mask]
                neg_label_path = os.path.join(self.neg_label_dir, str(lidar_token) + '_neg_label.bin')
                neg_label = np.fromfile(neg_label_path, dtype=np.uint8).reshape(-1, self.num_classes)[pt_with_img_idx]

                # labeled_points = np.zeros_like(labels_, dtype=np.int32)
                # neg_label_mask = np.sum(neg_label, axis=-1) > 0
                # labeled_points[neg_label_mask] = 1
                # prop_label_mask = (prop_label != self.ignored_labels)
                # labeled_points[prop_label_mask] = 2
                # labeled_points[sparse_label_mask] = 3

                inds = np.argsort(-sparse_label)
                voxel = voxel[inds, :]
                pts_cp = pts_cp[inds, :]
                sparse_label = sparse_label[inds]
                prop_label = prop_label[inds]
                neg_label = neg_label[inds]
                masks = masks[:, inds]
                pixel_coordinates = pixel_coordinates[:, inds, :]
                labels_ = labels_[inds]

        if self.use_color:
            feats_color = fetch_color(camera_channel, pixel_coordinates, masks)
            feat_ = np.concatenate([pts_cp, feats_color], axis=1)
        else:
            feat_ = pts_cp

        # inds: voxel使用散列方程key=(x*y_max+y)*z_max+z后, np.unique(key)的结果, 也就是重复网格坐标只取第一个遇到的
        _, inds, inverse_map = sparse_quantize(voxel,
                                               return_index=True,
                                               return_inverse=True)

        voxel_full = voxel[inds]
        feat_full = feat_[inds]
        labels_full = labels_[inds]
        masks = masks[:, inds]
        pixel_coordinates = pixel_coordinates[:, inds, :]
        lidar = SparseTensor(feat_full, voxel_full)
        labels = SparseTensor(labels_full, voxel_full)
        labels_ = SparseTensor(labels_, voxel)
        inverse_map = SparseTensor(inverse_map, voxel)

        feed_dict = {
            'lidar': lidar,
            'targets': labels,
            "targets_mapped": labels_,
            "inverse_map": inverse_map,
            'images': camera_channel,
            "pixel_coordinates": pixel_coordinates,  # [6, N, 2]
            "masks": masks,  # [6, N]
            'lidar_token': lidar_token,
            "pt_with_img_idx": [pt_with_img_idx],
        }

        if self.split == 'train':
            feed_dict['sparse_label'] = SparseTensor(sparse_label[inds], voxel_full)
            feed_dict['prop_label'] = SparseTensor(prop_label[inds], voxel_full)
            feed_dict['neg_label'] = SparseTensor(neg_label[inds], voxel_full)

        return feed_dict

    @staticmethod
    def collate_fn(batch):
        if isinstance(batch[0], dict):
            ans_dict = {}
            for key in batch[0].keys():
                if key == "masks":
                    ans_dict[key] = [torch.from_numpy(sample[key]) for sample in batch]
                elif key == "pixel_coordinates":
                    ans_dict[key] = [torch.from_numpy(sample[key]).float() for sample in batch]
                elif key == 'sp_labels':
                    ans_dict[key] = [torch.from_numpy(sample[key]) for sample in batch]
                elif key == 'sp_labels_anchor':
                    ans_dict[key] = [torch.from_numpy(sample[key]) for sample in batch]
                elif key == 'expand_idx':
                    ans_dict[key] = [torch.from_numpy(sample[key]) for sample in batch]
                elif key == 'expand_label':
                    ans_dict[key] = [torch.from_numpy(sample[key]) for sample in batch]
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
        else:
            ans_dict = tuple()
            for i in range(len(batch[0])):
                key = batch[0][i]
                if isinstance(key, SparseTensor):
                    ans_dict += sparse_collate([sample[i] for sample in batch]),
                elif isinstance(key, np.ndarray):
                    ans_dict += torch.stack(
                        [torch.from_numpy(sample[i]) for sample in batch], dim=0),
                elif isinstance(key, torch.Tensor):
                    ans_dict += torch.stack([sample[i] for sample in batch],
                                            dim=0),
                elif isinstance(key, dict):
                    ans_dict += sparse_collate_fn([sample[i] for sample in batch]),
                else:
                    ans_dict += [sample[i] for sample in batch],
            return ans_dict
