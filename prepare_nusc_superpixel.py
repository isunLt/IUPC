import os

import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn
import torch.cuda
import torch.utils.data as data

from torchvision.transforms import transforms

from visualize_utils import visualize_pcd, visualize_img

import cv2 as cv
from nuscenes import NuScenes as NuScenes_devkit
from PIL import Image


class NuScenes(data.Dataset):

    CAM_CHANNELS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    IMAGE_SIZE = (900, 1600)

    def __init__(self, nusc, split, im_cr, superpixel_dir):
        self.nusc = nusc
        self.split = split
        self.ignored_labels = np.sort([0])
        self.input_image_size = [int(x * im_cr) for x in self.IMAGE_SIZE]
        self.transform = transforms.Compose([transforms.Resize(size=self.input_image_size)])
        self.superpixel_dir = superpixel_dir
        if not os.path.exists(self.superpixel_dir):
            os.makedirs(self.superpixel_dir)

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

    def _calculate_superpixel(self, image):

        ih, iw, ic = image.shape
        image = image.astype(np.uint8)

        im_yuv = cv.cvtColor(image, cv.COLOR_RGB2YUV)
        channels = cv.split(im_yuv)
        cv.equalizeHist(channels[0], channels[0])
        cv.merge(channels, im_yuv)
        images = cv.cvtColor(im_yuv, cv.COLOR_YUV2RGB)
        im_hsv = cv.cvtColor(images, cv.COLOR_RGB2HSV)
        seeds = cv.ximgproc.createSuperpixelSEEDS(iw, ih, ic, num_superpixels=1024, num_levels=5, prior=5,
                                                  histogram_bins=10, double_step=True)
        seeds.iterate(im_hsv, 10)
        mask_seeds = seeds.getLabelContourMask()
        label_seeds = seeds.getLabels()
        # number_seeds = seeds.getNumberOfSuperpixels()
        mask_inv_seeds = cv.bitwise_not(mask_seeds)
        img_seeds = cv.bitwise_and(image, image, mask=mask_inv_seeds)
        # visualize_img(img_seeds.astype(np.uint8))

        # return label_seeds, img_seeds
        return label_seeds

    def __getitem__(self, index):
        sample = self.sample[index]
        for idx, channel in enumerate(self.CAM_CHANNELS):
            cam_token = sample['data'][channel]
            cam_channel = self.nusc.get('sample_data', cam_token)
            im = Image.open(os.path.join(self.nusc.dataroot, cam_channel['filename'])).convert('RGB')
            superpixel_path = os.path.join(self.superpixel_dir, str(cam_token) + '_superpixel.bin')
            if not os.path.exists(superpixel_path):
                # print('detect missing superpixel segment file', superpixel_path)
                superpixel_label = self._calculate_superpixel(np.array(self.transform(im))).astype(np.int32)
                superpixel_label.tofile(superpixel_path)
            else:
                superpixel_label = np.fromfile(superpixel_path, dtype=np.int32).reshape(self.input_image_size)

        return {
            'superpixel_label': superpixel_label
        }

    @staticmethod
    def collate_fn(batch):
        if isinstance(batch[0], dict):
            ans_dict = {}
            for key in batch[0].keys():
                ans_dict[key] = [sample[key] for sample in batch]
            return ans_dict
        else:
            ans_dict = tuple()
            for i in range(len(batch[0])):
                ans_dict += [sample[i] for sample in batch]
            return ans_dict


DATAROOT = '/data/stf/datasets/nuscenes'
IM_CR = 0.4
SUPERPIXEL_DIR = 'data/nuscenes/seeds_1024'
# SUPERPIXEL_DIR = '/data4/stf/iupc_publish/seeds_1024'

def main() -> None:

    nusc = NuScenes_devkit(dataroot=DATAROOT, version='v1.0-trainval', verbose=True)
    dataset = NuScenes(nusc=nusc, split='train', im_cr=IM_CR, superpixel_dir=SUPERPIXEL_DIR)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=32,
        shuffle=True,
        # shuffle=(split == 'train'),
        pin_memory=True,
        collate_fn=dataset.collate_fn)

    with torch.no_grad():
        for idx, feed_dict in enumerate(tqdm(dataloader)):
            sp = feed_dict['superpixel_label']

if __name__ == '__main__':
    main()
