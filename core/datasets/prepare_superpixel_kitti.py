import os
import os.path

import yaml
import numpy as np
from PIL import Image

from torchvision.transforms import transforms
from torchvision.transforms.functional import hflip, rotate, _get_inverse_affine_matrix, to_tensor, to_pil_image

import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn, sparse_collate
# from copy import deepcopy
import cv2 as cv

__all__ = ['SuperpixelSemanticKITTI']

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


class SuperpixelSemanticKITTI(dict):

    def __init__(self, root, voxel_size, **kwargs):
        config_path = os.path.join(root, 'semantic-kitti.yaml')
        with open(config_path, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        learning_map = semkittiyaml['learning_map']
        root = os.path.join(root, 'sequences')
        superpixel = kwargs.get('superpixel')
        sparse_label_root = kwargs.get('sparse_label_root')
        super().__init__({
            'train': _SuperpixelSemanticKITTIInternal(root, voxel_size, split='train', learning_map=learning_map,
                                                      superpixel=superpixel, sparse_label_root=sparse_label_root),
            'val': _SuperpixelSemanticKITTIInternal(root, voxel_size, split='val', learning_map=learning_map,
                                                    superpixel=superpixel, sparse_label_root=sparse_label_root)
        })


class _SuperpixelSemanticKITTIInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 split,
                 learning_map,
                 **kwargs):
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
        self.P_dict = {}
        self.Tr_dict = {}
        for seq in self.seqs:
            with open(os.path.join(self.root, seq, 'calib.txt'), 'r') as calib:
                P = []
                for idx in range(4):
                    line = calib.readline().rstrip('\n')[4:]
                    data = line.split(" ")
                    P.append(np.array(data, dtype=np.float32).reshape(3, -1))
                self.P_dict[seq] = P[2]
                line = calib.readline().rstrip('\n')[4:]
                data = line.split(" ")
                self.Tr_dict[seq] = np.array(data, dtype=np.float32).reshape((3, -1))

        self.pcd_files = []
        self.img_files = []
        self.map_idx2seq = []
        for seq in self.seqs:
            for pcd_name in sorted(os.listdir(os.path.join(self.root, seq, 'velodyne'))):
                self.pcd_files.append(os.path.join(self.root, seq, 'velodyne', str(pcd_name)))
                self.img_files.append(os.path.join(self.root, seq, 'image_2', str(pcd_name[:-4]) + '.png'))
                self.map_idx2seq.append(seq)
        self.IMAGE_SIZE = [368, 1224]  # 368, 1216
        # self.IMAGE_SIZE = [368, 1216]
        # self.CROP_SIZE = [368, 1216]
        # self.transform = transforms.CenterCrop(size=self.CROP_SIZE)
        self.transform = transforms.Resize(size=self.IMAGE_SIZE)
        self.augment = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4)  # not strengthened
            ], p=0.5),
            # transforms.RandomGrayscale(p=0.1)
        ])
        self.img_aug = False
        self.flip_aug = True
        self.flip_aug_rate = 0.5
        self.rotate_aug = False
        self.rotate_max_angle = [-15, 15]
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.superpixel_dir = kwargs.get('superpixel')
        self._mkdir(self.superpixel_dir)
        self.ignore_label = 0
        self.num_classes = 20
        self.sparse_label_root = kwargs.get('sparse_label_root')

    def __len__(self):
        # return 1
        return len(self.pcd_files)

    def _mkdir(self, dirpath):
        if not os.path.exists(dirpath):
            try:
                os.makedirs(dirpath)
            except OSError:
                print(f'Error occurred when creating ground truth mask dir "{dirpath}".')
            else:
                print(f'ground truth mask dir created "{dirpath}.')
        # else:
        #     print('%s exist!' % dirpath)

    def _load_pcd(self, index):
        filepath = self.pcd_files[index]
        pts = np.fromfile(filepath, dtype=np.float32).reshape((-1, 4))
        img = Image.open(self.img_files[index]).convert('RGB')
        if self.img_aug:
            img = self.augment(img)
        if self.split == 'test':
            labels_ = np.expand_dims(np.zeros_like(pts[:, 0], dtype=int), axis=1)
        else:
            lidar_label_path = filepath.replace('velodyne', 'labels')[:-3] + 'label'
            labels_ = np.fromfile(lidar_label_path, dtype=np.int32).reshape([-1, 1])
            labels_ = labels_ & 0xFFFF
            labels_ = np.vectorize(self.labels_mapping.__getitem__)(labels_).flatten()
        return pts, labels_, img

    def _mappcd2img(self, index, pts, im_size):
        seq = self.map_idx2seq[index]
        P, Tr = self.P_dict[seq], self.Tr_dict[seq]
        pts_homo = np.column_stack((pts, np.array([1] * pts.shape[0], dtype=pts.dtype)))
        Tr_homo = np.row_stack((Tr, np.array([0, 0, 0, 1], dtype=Tr.dtype)))
        pixel_coord = np.matmul(Tr_homo, pts_homo.T)
        pixel_coord = np.matmul(P, pixel_coord).T
        pixel_coord = pixel_coord / (pixel_coord[:, 2].reshape(-1, 1))
        pixel_coord = pixel_coord[:, :2]

        x_on_image = (pixel_coord[:, 0] >= 0) & (pixel_coord[:, 0] <= (im_size[0] - 1))
        y_on_image = (pixel_coord[:, 1] >= 0) & (pixel_coord[:, 1] <= (im_size[1] - 1))
        mask = x_on_image & y_on_image
        return pixel_coord, mask

    def _calculate_superpixel(self, image):

        ih, iw, ic = image.shape
        image = image.astype(np.uint8)

        im_yuv = cv.cvtColor(image, cv.COLOR_RGB2YUV)
        channels = cv.split(im_yuv)
        cv.equalizeHist(channels[0], channels[0])
        cv.merge(channels, im_yuv)
        images = cv.cvtColor(im_yuv, cv.COLOR_YUV2RGB)
        im_hsv = cv.cvtColor(images, cv.COLOR_RGB2HSV)
        seeds = cv.ximgproc.createSuperpixelSEEDS(iw, ih, ic, num_superpixels=2048, num_levels=5, prior=5,
                                                  histogram_bins=10, double_step=True)
        seeds.iterate(im_hsv, 10)
        mask_seeds = seeds.getLabelContourMask()
        label_seeds = seeds.getLabels()
        mask_inv_seeds = cv.bitwise_not(mask_seeds)
        img_seeds = cv.bitwise_and(image, image, mask=mask_inv_seeds)

        return label_seeds, img_seeds

    def _fetch_sp_labels(self, images: np.ndarray, pixel_coords: np.ndarray, masks: np.ndarray):
        _, h, w = images.shape
        sp_label = np.zeros(shape=(masks.shape[-1],), dtype=np.int32)
        for image, coord, mask in zip(images, pixel_coords, masks):
            coord = coord[mask, :]
            # coord[:, 0] = (coord[:, 0] + 1.0) / 2 * (w - 1.0)
            # coord[:, 1] = (coord[:, 1] + 1.0) / 2 * (h - 1.0)
            coord = np.floor(np.flip(coord, axis=-1)).astype(np.int32)
            sp_label[mask] = image[coord[:, 0], coord[:, 1]]

        return sp_label

    def _load_sparse_label(self, index, gt_label):
        token_list = self.pcd_files[index].split('/')
        seq, pcdname = token_list[-3], token_list[-1]
        sparse_label_path = os.path.join(self.sparse_label_root, str(seq), 'sparse_label', str(pcdname))
        sparse_label_mask = np.fromfile(sparse_label_path, dtype=np.uint8).astype(bool)
        sparse_label = np.full_like(gt_label, fill_value=self.ignore_label)
        sparse_label[sparse_label_mask] = gt_label[sparse_label_mask]
        prop_label_path = os.path.join(self.sparse_label_root, str(seq), 'prop_label', str(pcdname))
        prop_label = np.fromfile(prop_label_path, dtype=np.uint8)
        prop_label[sparse_label_mask] = sparse_label[sparse_label_mask]
        neg_label_path = os.path.join(self.sparse_label_root, str(seq), 'neg_label', str(pcdname))
        neg_label = np.fromfile(neg_label_path, dtype=np.uint8).reshape(-1, self.num_classes)
        return sparse_label, prop_label, neg_label

    # def _fetch_asso_target(self, sp_labels, sem_labels):
    #     m = sem_labels != self.ignore_label
    #     u_l = np.unique(sp_labels[m])
    #     tm = np.zeros_like(sp_labels, dtype=bool)
    #     for spl in u_l:
    #         tm = np.logical_or(tm, sp_labels == spl)
    #     return tm

    def _fetch_asa_label(self, sp_label, sem_label):
        m = sem_label != self.ignore_label
        u_l = np.unique(sp_label[m])
        tm = np.zeros_like(sp_label, dtype=np.uint8)
        for spl in u_l:
            mm = sp_label == spl
            sl = sem_label[mm]
            c = np.bincount(sl, minlength=self.num_classes)
            c[self.ignore_label] = 0
            cm = np.argmax(c)
            tm[mm] = cm
        return tm

    def _save_file(self, path, file: np.ndarray):

        if not os.path.exists(path):
            self._mkdir(os.path.dirname(path))
        file.astype(np.int32).tofile(path)

    def __getitem__(self, index):
        pts, labels_, img = self._load_pcd(index)
        sparse_label, prop_label, neg_label = self._load_sparse_label(index, labels_)
        img = np.array(img, dtype=np.uint8)
        token_list = self.pcd_files[index].split('/')
        seq, pcdname = token_list[-3], token_list[-1]
        superpixel_path = os.path.join(self.superpixel_dir, str(seq), str(pcdname))
        if not os.path.exists(superpixel_path):
            superpixel_channel, vis_image = self._calculate_superpixel(img)
            self._save_file(path=superpixel_path, file=superpixel_channel)
            # superpixel_channel.tofile(superpixel_path)
        else:
            superpixel_channel = np.fromfile(superpixel_path, dtype=np.int32).reshape_as(img)
        pts_ahead_idx = pts[:, 0] > 0
        pts = pts[pts_ahead_idx]
        labels_ = labels_[pts_ahead_idx]
        pixel_coordinates, mask = self._mappcd2img(index, pts[:, :3], (img.shape[1], img.shape[0]))
        sparse_label = sparse_label[pts_ahead_idx][mask]
        pts = pts[mask, :]
        labels_ = labels_[mask]
        pixel_coordinates = pixel_coordinates[mask, :]
        masks = mask[mask]
        # pixel_coordinates[:, 0] = pixel_coordinates[:, 0] / (img.size[0] - 1) * 2 - 1.0
        # pixel_coordinates[:, 1] = pixel_coordinates[:, 1] / (img.size[1] - 1) * 2 - 1.0
        pixel_coordinates = np.expand_dims(pixel_coordinates, axis=0)
        masks = np.expand_dims(masks, axis=0)
        superpixel_channel = np.expand_dims(superpixel_channel, axis=0)
        sp_label = self._fetch_sp_labels(superpixel_channel, pixel_coordinates, masks)
        asa_label = self._fetch_asa_label(sp_label, sparse_label)

        return {
            'targets': labels_,
            'asa_labels': asa_label,
            'vis_image': vis_image
        }

    @staticmethod
    def collate_fn(batch):
        ans_dict = {}
        for key in batch[0].keys():
            ans_dict[key] = [sample[key] for sample in batch]
        return ans_dict
