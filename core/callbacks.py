import os
import numpy as np
from typing import Any, Dict, Optional

import torch

from torchpack.environ import get_run_dir
from torchpack import distributed as dist
from torchpack.callbacks import TFEventWriter
from torchpack.callbacks.callback import Callback
from torchpack.utils import fs, io
from torchpack.utils.logging import logger
from nuscenes.eval.lidarseg.utils import ConfusionMatrix

from visualize_utils import draw_bar_chart, draw_confuse_matrix, SemKITTI_label_name_16, SemKITTI_label_name_19, SemKITTI_label_name_22

from prettytable import PrettyTable

__all__ = ['MeanIoU', 'EpochSaver', 'PseudoLabelEvaluator', 'PseudoLabelSaver', 'PseudoLabelVisualizer', 'SemKITTIPseudoLabelSaver']


class MeanIoU(Callback):
    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou') -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.mapclass2name = None
        if self.num_classes == 17:
            self.mapclass2name = SemKITTI_label_name_16
        elif self.num_classes == 20:
            self.mapclass2name = SemKITTI_label_name_19
        elif self.num_classes == 23:
            self.mapclass2name = SemKITTI_label_name_22

    def _before_epoch(self) -> None:
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        if type(outputs) != np.ndarray:
            for i in range(self.num_classes):
                self.total_seen[i] += torch.sum(targets == i).item()
                self.total_correct[i] += torch.sum(
                    (targets == i) & (outputs == targets)).item()
                self.total_positive[i] += torch.sum(
                    outputs == i).item()
        else:
            for i in range(self.num_classes):
                self.total_seen[i] += np.sum(targets == i)
                self.total_correct[i] += np.sum((targets == i)
                                                & (outputs == targets))
                self.total_positive[i] += np.sum(outputs == i)

    def _after_epoch(self) -> None:
        for i in range(self.num_classes):
            self.total_seen[i] = dist.allreduce(self.total_seen[i], reduction='sum')
            self.total_correct[i] = dist.allreduce(self.total_correct[i], reduction='sum')
            self.total_positive[i] = dist.allreduce(self.total_positive[i], reduction='sum')

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                if i == self.ignore_label:
                    continue
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i] +
                                                   self.total_positive[i] -
                                                   self.total_correct[i])
                ious.append(cur_iou)

        miou = np.mean(ious)
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, miou * 100)
            for writer in self.trainer.summary.writers:
                if isinstance(writer, TFEventWriter):
                    for idx in range(1, self.num_classes):
                        writer.add_scalar(self.name + '/' + self.mapclass2name[idx], ious[idx-1] * 100)
            pt = PrettyTable()
            pt.field_names = ['Item'] + list(self.mapclass2name.values())[1:] + ['Mean']
            pt.add_row(['IoU'] + [round(i * 100, 2) for i in ious] + [round(miou * 100, 2)])
            print(pt)
        else:
            pt = PrettyTable()
            pt.field_names = ['Item'] + list(self.mapclass2name.values())[1:] + ['Mean']
            pt.add_row(['IoU'] + [round(i * 100, 2) for i in ious] + [round(miou * 100, 2)])
            print(pt)


class EpochSaver(Callback):
    """
    Save the checkpoint once triggered.
    """
    master_only: bool = True

    def __init__(self, *, epoch_to_save: int = 5,
                 save_dir: Optional[str] = None) -> None:
        self.epoch_to_save = epoch_to_save
        if save_dir is None:
            save_dir = os.path.join(get_run_dir(), 'checkpoints')
        self.save_dir = fs.normpath(save_dir)

    def _trigger_epoch(self) -> None:
        self._trigger()

    def _trigger(self) -> None:
        if self.trainer.epoch_num and not (self.trainer.epoch_num % self.epoch_to_save):
            save_path = os.path.join(self.save_dir,
                                     f'epoch-{self.trainer.epoch_num}.pt')
            try:
                io.save(save_path, self.trainer.state_dict())
            except OSError:
                logger.exception(
                    f'Error occurred when saving checkpoint "{save_path}".')
            else:
                logger.info(f'Checkpoint saved: "{save_path}".')


class PseudoLabelSaver(Callback):

    def __init__(self, save_dir: Optional[str] = None):

        if save_dir is None:
            c_it = 0
            for d in os.listdir(get_run_dir()):
                if 'pseudo_labels' in d:
                    c_it += 1
            save_dir = os.path.join(get_run_dir(), 'pseudo_labels_' + str(c_it))

        self.save_dir = fs.normpath(save_dir)

    def _before_train(self) -> None:

        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except OSError:
                logger.exception(f'Error occurred when creating pseudo label folder "{self.save_dir}".')
            else:
                logger.info(f'pseudo label folder created "{self.save_dir}.')
        else:
            logger.info(f'use pseudo label folder "{self.save_dir}.')

    def _after_step(self, output_dict: Dict[str, Any]) -> None:

        pseudo_label_list = output_dict.get('pseudo_label_list')
        lidar_token_list = output_dict.get('lidar_token_list')

        try:
            for pseudo_label, lidar_token in zip(pseudo_label_list, lidar_token_list):
                pseudo_label = pseudo_label.cpu().numpy().astype(np.uint8)
                save_path = os.path.join(self.save_dir, str(lidar_token) + '_pseudo_label.bin')
                pseudo_label.tofile(save_path)
        except OSError:
            logger.exception(f'Error occured when saving pseudo label "{lidar_token}".')


class PseudoLabelEvaluator(Callback):
    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 xbar_names: list = None) -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.save_dir = os.path.join(get_run_dir(), 'figures')
        self.xbar_names = xbar_names if xbar_names is not None else list(SemKITTI_label_name_16.values())[1:]
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except OSError:
                logger.exception(f'Error occurred when creating folder "{self.save_dir}".')
            else:
                logger.info(f'folder created "{self.save_dir}.')
        c_it = 0
        for d in os.listdir(self.save_dir):
            if 'pseudo_label_eval' in d:
                c_it += 1
        self.figure_save_path = os.path.join(self.save_dir, 'pseudo_label_eval_' + str(c_it))
        self.text_save_path = os.path.join(self.save_dir, 'pseudo_label_accuracy_' + str(c_it) + '.txt')

    def _before_epoch(self) -> None:
        self.total_num_per_class = torch.zeros(size=[self.num_classes, ], dtype=torch.float)
        self.correct_num_per_class = torch.zeros_like(self.total_num_per_class)
        self.fault_num_per_class = torch.zeros_like(self.total_num_per_class)
        self.ignore_num_per_class = torch.zeros_like(self.total_num_per_class)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:

        ground_truth_list = output_dict.get('ground_truth_list')
        pseudo_label_list = output_dict.get('pseudo_label_list')

        for gt, pl in zip(ground_truth_list, pseudo_label_list):
            valid_mask = gt != self.ignore_label
            gt, pl = gt[valid_mask], pl[valid_mask]
            self.total_num_per_class += torch.bincount(gt, minlength=self.num_classes)
            self.correct_num_per_class += torch.bincount(gt[gt == pl], minlength=self.num_classes)
            pl_valid_mask = pl != self.ignore_label
            self.fault_num_per_class += torch.bincount(gt[pl_valid_mask][gt[pl_valid_mask] != pl[pl_valid_mask]], minlength=self.num_classes)
            self.ignore_num_per_class += torch.bincount(gt[~pl_valid_mask], minlength=self.num_classes)

    def _after_epoch(self) -> None:

        correct_ratio_per_class = self.correct_num_per_class / self.total_num_per_class
        fault_ratio_per_class = self.fault_num_per_class / self.total_num_per_class
        ignore_ratio_per_class = self.ignore_num_per_class / self.total_num_per_class

        draw_bar_chart(
            bar_val_list=[correct_ratio_per_class[1:], fault_ratio_per_class[1:], ignore_ratio_per_class[1:]],
            bar_name_list=self.xbar_names,
            # bar_name_list=list(SemKITTI_label_name_16.values())[1:],
            col_name_list=['correct_ratio', 'fault_ratio', 'ignore_ratio'],
            fig_save_path=self.figure_save_path)

        logger.info(f'figure saves to {self.figure_save_path}')

        with open(self.text_save_path, 'a') as f:
            # f.write(str(list(SemKITTI_label_name_16.values())[1:]) + '\n')
            f.write(str(self.xbar_names) + '\n')
            f.write('correct: ' + str(correct_ratio_per_class[1:].cpu().numpy()) + '\n')
            f.write('fault:' + str(fault_ratio_per_class[1:].cpu().numpy()) + '\n')
            f.write('ignore:' + str(ignore_ratio_per_class[1:].cpu().numpy()) + '\n')
            f.write('mean correct:' + str(torch.mean(correct_ratio_per_class[1:]).cpu().numpy()) + '\n')
            f.write('mean fault:' + str(torch.mean(fault_ratio_per_class[1:]).cpu().numpy()) + '\n')
            f.write('mean ignore:' + str(torch.mean(ignore_ratio_per_class[1:]).cpu().numpy()) + '\n')

        logger.info(f'txt file saves to {self.figure_save_path}')


class ConfuseMatrix(Callback):
    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 fig_size: tuple = (6.4, 4.8),
                 xbar_names: list = None) -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.fig_size = fig_size
        self.save_dir = os.path.join(get_run_dir(), 'figures')
        self.xbar_names = xbar_names if xbar_names is not None else list(SemKITTI_label_name_16.values())[1:]
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except OSError:
                logger.exception(f'Error occurred when creating folder "{self.save_dir}".')
            else:
                logger.info(f'folder created "{self.save_dir}.')
        c_it = 0
        for d in os.listdir(self.save_dir):
            if 'pseudo_label_eval' in d:
                c_it += 1
        self.figure_save_path = os.path.join(self.save_dir, 'pseudo_label_confuse_matrix_' + str(c_it))

    def _before_epoch(self) -> None:
        self.confuse_matrix = ConfusionMatrix(num_classes=self.num_classes, ignore_idx=self.ignore_label)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:

        ground_truth_list = output_dict.get('ground_truth_list')
        pseudo_label_list = output_dict.get('pseudo_label_list')

        for gt, pl in zip(ground_truth_list, pseudo_label_list):
            pl_valid_mask = pl != self.ignore_label
            if isinstance(gt, torch.Tensor):
                gt = gt.detach().cpu().numpy()
            if isinstance(pl, torch.Tensor):
                pl = pl.detach().cpu().numpy()
            self.confuse_matrix.update(gt_array=gt[pl_valid_mask], pred_array=pl[pl_valid_mask])

    def _after_epoch(self) -> None:

        c_m = self.confuse_matrix.global_cm
        c_m[self.ignore_label, :] = 0
        c_m[:, self.ignore_label] = 0

        draw_confuse_matrix(bar_name_list=self.xbar_names, confuse_matrix=c_m[1:, 1:],
                            fig_save_path=self.figure_save_path, normalize=True, fig_size=self.fig_size)

        logger.info(f'figure saves to {self.figure_save_path}')


class PseudoLabelVisualizer(Callback):
    def __init__(self, dataset) -> None:
        from visualize_utils import visualize_pcd
        self.visualizer = visualize_pcd
        self.dataset = dataset
        if dataset == 'semkitti':
            from visualize_utils import MapSemKITTI2NUSC
            self.label_mapper = MapSemKITTI2NUSC
        elif dataset == 'waymo':
            from visualize_utils import MapWaymo2NUSC
            self.label_mapper = MapWaymo2NUSC
        else:
            self.label_mapper = None

    def _load_pcd(self, path):
        if self.dataset == 'nusc':
            return np.fromfile(path, dtype=np.float32).reshape([-1, 5])[:, :3]
        elif self.dataset == 'semkitti':
            return np.fromfile(path, dtype=np.float32).reshape([-1, 4])[:, :3]
        elif self.dataset == 'waymo':
            return np.fromfile(path, dtype=np.float32).reshape([-1, 6])[:, :3]
        else:
            return np.fromfile(path, dtype=np.float32).reshape([-1, 3])

    def _label_mapping(self, label):
        return torch.tensor([self.label_mapper[int(l.item())] for l in label], dtype=torch.long, device=label.device)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:

        ground_truth_list = output_dict.get('ground_truth_list')
        pseudo_label_list = output_dict.get('pseudo_label_list')
        lidar_path_list = output_dict.get('lidar_path_list')
        neg_masks = output_dict.get('neg_masks', None)

        for idx, (lidar_path, gt, pl) in enumerate(zip(lidar_path_list, ground_truth_list, pseudo_label_list)):
            pts = self._load_pcd(lidar_path)
            if self.label_mapper:
                pl = self._label_mapping(pl)
                gt = self._label_mapping(gt)
            self.visualizer(pts[neg_masks[idx]], predict=pl, target=gt)


class SemKITTIPseudoLabelSaver(Callback):

    def __init__(self, save_dir: Optional[str] = None):

        if save_dir is None:
            c_it = 0
            for d in os.listdir(get_run_dir()):
                if 'pseudo_labels' in d:
                    c_it += 1
            save_dir = os.path.join(get_run_dir(), 'pseudo_labels_' + str(c_it))

        self.save_dir = fs.normpath(save_dir)

    def _before_train(self) -> None:

        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except OSError:
                logger.exception(f'Error occurred when creating pseudo label folder "{self.save_dir}".')
            else:
                logger.info(f'pseudo label folder created "{self.save_dir}.')

    def _after_step(self, output_dict: Dict[str, Any]) -> None:

        pseudo_label_list = output_dict.get('pseudo_label_list')
        lidar_path_list = output_dict.get('lidar_path_list')

        try:
            for pseudo_label, lidar_path in zip(pseudo_label_list, lidar_path_list):
                pseudo_label = pseudo_label.cpu().numpy().astype(np.uint8)
                token_list = lidar_path.split('/')
                seq, pcdname = token_list[-3], token_list[-1]
                save_path = os.path.join(self.save_dir, str(seq), str(pcdname))
                dirname = os.path.dirname(save_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                pseudo_label.tofile(save_path)
        except OSError:
            logger.exception(f'Error occured when saving pseudo label "{lidar_path}".')


class WaymoPseudoLabelSaver(Callback):

    def __init__(self, save_dir: Optional[str] = None):

        if save_dir is None:
            c_it = 0
            for d in os.listdir(get_run_dir()):
                if 'pseudo_labels' in d:
                    c_it += 1
            save_dir = os.path.join(get_run_dir(), 'pseudo_labels_' + str(c_it))

        self.save_dir = fs.normpath(save_dir)

    def _before_train(self) -> None:

        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except OSError:
                logger.exception(f'Error occurred when creating pseudo label folder "{self.save_dir}".')
            else:
                logger.info(f'pseudo label folder created "{self.save_dir}.')

    def _after_step(self, output_dict: Dict[str, Any]) -> None:

        pseudo_label_list = output_dict.get('pseudo_label_list')
        lidar_path_list = output_dict.get('lidar_path_list')

        try:
            for pseudo_label, lidar_path in zip(pseudo_label_list, lidar_path_list):
                pseudo_label = pseudo_label.cpu().numpy().astype(np.uint8)
                token_list = lidar_path.split('/')
                seq, pcdname = token_list[-3], token_list[-1]
                save_path = os.path.join(self.save_dir, str(seq), str(pcdname))
                dirname = os.path.dirname(save_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                pseudo_label.tofile(save_path)
        except OSError:
            logger.exception(f'Error occured when saving pseudo label "{lidar_path}".')

class PredictionSaver(Callback):

    def __init__(self, save_dir: str):
        self.save_dir = fs.normpath(save_dir)

    def _before_train(self) -> None:

        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except OSError:
                logger.exception(f'Error occurred when creating global confidence folder "{self.save_dir}".')
            else:
                logger.info(f'prediction folder created "{self.save_dir}.')

    def _after_step(self, output_dict: Dict[str, Any]) -> None:

        outputs_vox = output_dict.get('outputs_vox').detach().cpu().numpy().astype(np.uint8)
        lidar_token_list = output_dict.get('lidar_token_list')
        num_pts = output_dict.get('num_pts')
        cur = 0
        try:
            for i, n in enumerate(num_pts):
                pred = outputs_vox[cur:cur+n]
                save_path = os.path.join(self.save_dir, str(lidar_token_list[i]) + '_pred.bin')
                pred.tofile(save_path)
                cur += n
        except OSError:
            logger.exception(f'Error occured when saving pseudo label "{lidar_token_list[i]}".')


class SemKITTIPredictionSaver(Callback):

    def __init__(self, save_dir: str):
        self.save_dir = fs.normpath(save_dir)

    def _before_train(self) -> None:

        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except OSError:
                logger.exception(f'Error occurred when creating prediction folder "{self.save_dir}".')
            else:
                logger.info(f'prediction folder created "{self.save_dir}.')

    def _after_step(self, output_dict: Dict[str, Any]) -> None:

        outputs_vox = output_dict.get('outputs_vox').detach().cpu().numpy().astype(np.uint8)
        lidar_token_list = output_dict.get('lidar_token_list')
        num_pts = output_dict.get('num_pts')
        cur = 0
        try:
            for i, n in enumerate(num_pts):
                pred = outputs_vox[cur:cur+n]
                lidar_path = lidar_token_list[i]
                token_list = lidar_path.split('/')
                seq, pcdname = token_list[-3], token_list[-1]
                save_path = os.path.join(self.save_dir, str(seq), str(pcdname))
                dirname = os.path.dirname(save_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                pred.tofile(save_path)
                cur += n
        except OSError:
            logger.exception(f'Error occured when saving pseudo label "{lidar_token_list[i]}".')