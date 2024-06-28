import os
import numpy as np
from functools import reduce

import torch
from torch.utils import data

from nuscenes import NuScenes as NuScenes_devkit
from nuscenes.utils.data_classes import transform_matrix

from torchpack.environ import get_run_dir

from core.datasets.utils import PCDTransformTool
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn, sparse_collate
from pyquaternion import Quaternion

import hdbscan
import open3d as o3d
from torchpack.utils.config import configs



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


class NuScenesSparseLabelPrepare(dict):
    def __init__(self, root, version, verbose):
        self.nusc = NuScenes_devkit(dataroot=root, version=version, verbose=verbose)
        super(NuScenesSparseLabelPrepare, self).__init__({
            "train": _NuScenesSparseLabelPrepare(nusc=self.nusc, split="train")
        })


class _NuScenesSparseLabelPrepare(data.Dataset):
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

    def __init__(self, nusc, split="train"):
        self.nusc = nusc
        self.split = split
        self.ignored_labels = np.sort([0])

        self.run_dir = get_run_dir()
        self.weak_label_path = configs['dataset']['sparse_label_path']
        self.sparse_label_dir = os.path.join(self.weak_label_path, 'sparse_label')
        self.prop_label_dir = os.path.join(self.weak_label_path, 'prop_label')
        self.neg_label_dir = os.path.join(self.weak_label_path, 'neg_label')
        self.num_classes = configs['data']['num_classes']

        # plane segmentation and cluster hyperparameters
        if configs['dataset']['clusterer']['name'] == 'hdbscan':
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=configs['dataset']['clusterer']['min_cluster_size'],
                cluster_selection_epsilon=configs['dataset']['clusterer']['cluster_selection_epsilon']
            )
        else:
            raise  NotImplementedError
        self.grid_size = np.array(configs['dataset']['grid_size'])
        self.grid_size[1] *= np.pi
        self.ps_min_num = configs['dataset']['ps_min_num']
        self.ps_norm_th = configs['dataset']['ps_norm_th']
        self.ps_dis_th = configs['dataset']['ps_dis_th']
        self.ps_outlier_th = configs['dataset']['ps_outlier_th']
        self.point_per_class = configs['dataset']['point_per_class']
        self.thing_class = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]).astype(bool)

        self._mkdir(self.sparse_label_dir)
        self._mkdir(self.prop_label_dir)
        self._mkdir(self.neg_label_dir)

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

    def _aggregate_lidar_sweeps(self, sample_ref, nsweeps=5, with_next=False):

        def _remove_close(xyz: np.ndarray, min_dist):
            x_mask = np.fabs(xyz[:, 0]) < min_dist
            y_mask = np.fabs(xyz[:, 1]) < min_dist
            return np.logical_and(x_mask, y_mask)

        # Get reference pose and timestamp.
        ref_sd_token = sample_ref['data']['LIDAR_TOP']
        ref_sd_rec = self.nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = self.nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)

        # Aggregate current and previous sweeps.
        current_sd_rec = ref_sd_rec
        pts = []
        ts = []
        for _ in range(nsweeps):
            if current_sd_rec['prev'] == '':
                break
            current_sd_rec = self.nusc.get('sample_data', current_sd_rec['prev'])
            # Load up the pointcloud and remove points close to the sensor.
            curr_pts_path = os.path.join(self.nusc.dataroot, current_sd_rec['filename'])
            curr_pts = np.fromfile(curr_pts_path, dtype=np.float32).reshape([-1, 5])[:, :3]
            close_mask = _remove_close(curr_pts, min_dist=1.0)
            curr_pts = curr_pts[~close_mask]
            pcd_trans_tool = PCDTransformTool(curr_pts)

            # Get past pose.
            current_pose_rec = self.nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            pcd_trans_tool.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
            times = time_lag * np.ones((curr_pts.shape[0],))
            ts.append(times)

            # Merge with key pc.
            pts.append(pcd_trans_tool.pcd.T)

        current_sd_rec = ref_sd_rec
        if with_next:
            for _ in range(nsweeps):
                if current_sd_rec['next'] == '':
                    break
                current_sd_rec = self.nusc.get('sample_data', current_sd_rec['next'])
                # Load up the pointcloud and remove points close to the sensor.
                curr_pts_path = os.path.join(self.nusc.dataroot, current_sd_rec['filename'])
                curr_pts = np.fromfile(curr_pts_path, dtype=np.float32).reshape([-1, 5])[:, :3]
                close_mask = _remove_close(curr_pts, min_dist=1.0)
                curr_pts = curr_pts[~close_mask]
                pcd_trans_tool = PCDTransformTool(curr_pts)

                # Get past pose.
                current_pose_rec = self.nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(current_pose_rec['translation'],
                                                   Quaternion(current_pose_rec['rotation']), inverse=False)

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
                car_from_current = transform_matrix(current_cs_rec['translation'],
                                                    Quaternion(current_cs_rec['rotation']),
                                                    inverse=False)

                # Fuse four transformation matrices into one and perform transform.
                trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
                pcd_trans_tool.transform(trans_matrix)

                # Add time vector which can be used as a temporal feature.
                time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
                times = time_lag * np.ones((curr_pts.shape[0],))
                ts.append(times)

                # Merge with key pc.
                pts.append(pcd_trans_tool.pcd.T)

        return pts, ts

    def _cat2polar(self, xyz: np.ndarray):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2  # x^2+y^2
        ptsnew[:, 3] = np.sqrt(xy)
        ptsnew[:, 4] = np.arctan2(xyz[:, 1], xyz[:, 0])
        ptsnew[:, 5] = xyz[:, 2]
        return ptsnew[:, 3:6]

    def _polar_grid(self, xyz: np.ndarray):
        loc_polar = self._cat2polar(xyz)
        grid = np.round(loc_polar[:, :2] / self.grid_size.reshape([1, 2])).astype(np.int32)
        grid -= grid.min(0, keepdims=True)
        max_angle = np.round(2 * np.pi / self.grid_size[1]).astype(np.int32)
        m = (grid[:, 1] == max_angle)
        grid[m, 1] = 0
        return grid

    def _filter_ground_points(self, xyz: np.ndarray, coord):

        def _fit_plane_ransac(xyz: np.ndarray, dis_th=0.2, r_n=10, it_num=1000):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])

            plane_model, inliers = pcd.segment_plane(distance_threshold=dis_th,
                                                     ransac_n=r_n,
                                                     num_iterations=it_num)
            return plane_model, inliers

        def _check_norm(plane_model, coord, norm_th=0.5):
            norm_vec = plane_model[:3]
            norm_vec /= np.linalg.norm(norm_vec)
            z_axis = np.array([0., 0., 1.])
            angle = np.fabs(np.dot(norm_vec, z_axis))
            if coord[0] < 2 and (4 <= coord[1] <= 5):
                return angle > 0.95
            return angle > norm_th

        def _check_barycenter(xyz: np.ndarray, inlier, outline_th=0.1):
            outline_num = np.round(xyz.shape[0] * outline_th).astype(np.int32)
            if np.sum(~inlier) <= outline_num:
                return True
            inliner_z = np.mean(xyz[inlier][:, 2])
            outliner_z = np.mean(xyz[~inlier][:, 2])
            return inliner_z <= outliner_z

        def _check_fitness(inlier, valid_num, flag):
            # return True
            return flag or (np.sum(inlier) / float(valid_num) >= 0.5)

        def _get_inlier(xyz: np.ndarray, plane_model, dis_th=0.2):
            xyz_homo = np.hstack((xyz, np.ones(shape=(xyz.shape[0], 1))))
            dist = np.fabs(np.sum(xyz_homo * plane_model.reshape([1, 4]), axis=-1)) / np.linalg.norm(plane_model[:3])
            return dist <= dis_th

        m = np.ones(shape=(xyz.shape[0],), dtype=bool)
        if np.std(xyz[:, 2]) >= 0.75:
            z_mask = xyz[:, 2] > np.mean(xyz[:, 2])
            m[z_mask] = False
        it_c = 3
        valid_num = np.sum(m)
        flag = False
        while np.sum(m) > self.ps_min_num and it_c:
            plane_model, _ = _fit_plane_ransac(xyz[m], dis_th=self.ps_dis_th)
            inlier = _get_inlier(xyz=xyz, plane_model=plane_model, dis_th=self.ps_dis_th)
            if not _check_fitness(inlier=inlier, valid_num=valid_num, flag=flag):
                break
            if _check_norm(plane_model, coord=coord, norm_th=self.ps_norm_th) and \
                    _check_barycenter(xyz=xyz, inlier=inlier, outline_th=self.ps_outlier_th):
                return inlier
            else:
                m[inlier] = False
                it_c -= 1
                flag = True

        return np.zeros_like(m, dtype=bool)

    def _mkdir(self, dirpath):
        if not os.path.exists(dirpath):
            try:
                os.makedirs(dirpath)
            except OSError:
                print(f'Error occurred when creating ground truth mask dir "{dirpath}".')
            else:
                print(f'ground truth mask dir created "{dirpath}.')

    def _save_files(self, lidar_token, feed_dict: dict):

        def _save_file(fov_mask, file: np.ndarray, filepath):
            l = np.zeros_like(fov_mask, dtype=np.uint8)
            l[fov_mask] = file.astype(np.uint8)
            l.tofile(filepath)

        def _save_neg_label(fov_mask, file: np.ndarray, filepath):
            l = np.zeros((fov_mask.shape[0], file.shape[1]), dtype=np.uint8)
            l[fov_mask] = file.astype(np.uint8)
            l.tofile(filepath)

        sparse_label_path = os.path.join(self.sparse_label_dir, str(lidar_token) + '_sparse_label.bin')
        prop_label_path = os.path.join(self.prop_label_dir, str(lidar_token) + '_prop_label.bin')
        neg_label_path = os.path.join(self.neg_label_dir, str(lidar_token) + '_neg_label.bin')
        sparse_label = feed_dict.get('sparse_label_mask', None)
        if not os.path.exists(sparse_label) and sparse_label is not None:
            _save_file(feed_dict['fov_mask'], sparse_label, sparse_label_path)
        prop_label = feed_dict.get('prop_label_mask', None)
        if not os.path.exists(prop_label) and prop_label is not None:
            _save_file(feed_dict['fov_mask'], prop_label, prop_label_path)
        neg_label = feed_dict.get('neg_label_mask', None)
        if not os.path.exists(neg_label) and neg_label is not None:
            _save_neg_label(feed_dict['fov_mask'], neg_label, neg_label_path)

    def __getitem__(self, index):
        sample = self.sample[index]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_channel = self.nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_channel["filename"])
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])[:, :3]  # N, 3

        valid_mask = np.array([-1] * pts.shape[0])
        for idx, channel in enumerate(self.CAM_CHANNELS):
            cam_token = sample['data'][channel]
            cam_channel = self.nusc.get('sample_data', cam_token)
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
            pixel_coord[0, :] = pixel_coord[0, :] / (self.IMAGE_SIZE[1] - 1.0) * 2.0 - 1.0  # width
            pixel_coord[1, :] = pixel_coord[1, :] / (self.IMAGE_SIZE[0] - 1.0) * 2.0 - 1.0  # height
            # pixel_coordinates.append(pixel_coord.T)

            # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
            # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            mask = np.logical_and(mask, pixel_coord[0, :] > -1)
            mask = np.logical_and(mask, pixel_coord[0, :] < 1)
            mask = np.logical_and(mask, pixel_coord[1, :] > -1)
            mask = np.logical_and(mask, pixel_coord[1, :] < 1)
            valid_mask[mask] = idx

        fov_mask = (valid_mask != -1)
        # fov_mask = np.ones(shape=[pts.shape[0],], dtype=bool)
        pts = pts[fov_mask]

        lidar_label_path = os.path.join(self.nusc.dataroot, self.nusc.get("lidarseg", lidar_token)["filename"])
        labels_ = np.fromfile(lidar_label_path, dtype=np.uint8).reshape([-1, 1])
        labels_ = np.vectorize(self.labels_mapping.__getitem__)(labels_).flatten()
        labels_ = labels_[fov_mask]

        agg_pts, agg_ts = self._aggregate_lidar_sweeps(sample_ref=sample, nsweeps=5, with_next=False)
        agg_pts = np.concatenate([pts] + agg_pts, axis=0)
        agg_ts = np.concatenate([np.zeros(pts.shape[0], dtype=np.float32)] + agg_ts, axis=0)
        agg_labels = np.zeros((agg_pts.shape[0],), dtype=np.uint8)
        agg_labels[:labels_.shape[0]] = labels_
        grid = self._polar_grid(xyz=agg_pts)
        uq, inv = np.unique(grid, axis=0, return_inverse=True)

        key_frame_mask = agg_ts == 0
        assert np.sum(key_frame_mask) == labels_.shape[0]
        ground_mask = np.zeros(shape=(agg_pts.shape[0],), dtype=bool)
        sparse_label_mask = np.zeros_like(labels_, dtype=bool)
        prop_label_mask = np.zeros_like(labels_, dtype=np.uint8)
        neg_label_mask = np.zeros(shape=(labels_.shape[0], self.num_classes), dtype=np.uint8)

        # label the ground
        for uq_i in uq:
            m = np.all(grid == uq_i, axis=-1)
            m = np.nonzero(m)[0]
            inlier = self._filter_ground_points(xyz=agg_pts[m], coord=uq_i)
            ground_mask[m[inlier]] = True

            g_idx = m[inlier]
            g_idx = g_idx[g_idx < labels_.shape[0]]
            g_l = labels_[g_idx]
            uq_l = np.unique(g_l)
            if uq_l.shape[0] == 1 and uq_l[0] == self.ignored_labels:
                continue
            for c in uq_l:
                c_m = np.nonzero(g_l == c)[0]
                s_n = min(c_m.shape[0], self.point_per_class)
                if c in self.ignored_labels or uq_l.shape[0] == 1:
                    s_n = 1
                c_idx = np.random.choice(c_m, (s_n,), replace=False)
                sparse_label_mask[g_idx[c_idx]] = True
            if uq_l.shape[0] == 1:
                prop_label_mask[g_idx] = uq_l[0]
            else:
                for c in uq_l:
                    neg_label_mask[g_idx, c] = 1

        resume_mask = np.nonzero(np.logical_and(key_frame_mask, ~ground_mask))[0]
        pts_front = pts[resume_mask]
        label_front = labels_[resume_mask]

        self.clusterer.fit(pts_front)
        front_cluster = self.clusterer.labels_

        # label others
        uq = np.unique(front_cluster)
        for uq_i in uq:
            if uq_i == -1:
                continue
            m = np.nonzero(front_cluster == uq_i)[0]
            m_l = label_front[m]
            uq_l, uq_c = np.unique(m_l, return_counts=True)
            if uq_l.shape[0] == 1 and uq_l[0] == self.ignored_labels:
                continue
            # we mimic human annotators by ignoring small clusters
            if np.sum(self.thing_class[uq_l]) == 0:
                if 15 in uq_l:
                    if m.shape[0] < 10:
                        continue
                elif m.shape[0] < 20:
                    continue
            else:
                if m.shape[0] < 5:
                    continue
            if np.sum(self.thing_class[uq_l]) == 0:
                minor_mask = (uq_c / np.sum(uq_c)) <= 0.01  # 0.05
            elif np.sum(self.thing_class[uq_l]) == 1:
                ci = np.argmax(uq_c)
                if self.thing_class[uq_l[ci]]:
                    minor_mask = (uq_c / np.sum(uq_c)) <= 0.1
                else:
                    minor_mask = (uq_c / np.sum(uq_c)) <= 0.001
            else:
                minor_mask = (uq_c / np.sum(uq_c)) <= 0.001
            uq_l = uq_l[~minor_mask]
            for c in uq_l:
                c_m = np.nonzero(m_l == c)[0]
                s_n = min(c_m.shape[0], self.point_per_class)
                if c in self.ignored_labels or uq_l.shape[0] == 1:
                    s_n = 1
                c_idx = np.random.choice(c_m, (s_n,), replace=False)
                sparse_label_mask[resume_mask[m[c_idx]]] = True
            if uq_l.shape[0] == 1:
                prop_label_mask[resume_mask[m]] = uq_l[0]
            else:
                for c in uq_l:
                    neg_label_mask[resume_mask[m], c] = 1

        feed_dict = {
            'pts': pts,
            'targets': labels_,
            'sparse_label_mask': sparse_label_mask,
            'prop_label_mask': prop_label_mask,
            'neg_label_mask': neg_label_mask,
            'fov_mask': fov_mask,
            'lidar_token': lidar_token
        }

        self._save_files(lidar_token, feed_dict)

        return feed_dict


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
