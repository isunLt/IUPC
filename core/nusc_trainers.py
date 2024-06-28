import os
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
from torch import nn
from torch.cuda import amp
import torch.nn.functional as F

from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
from torchpack.callbacks import Callback, ProgressBar
from torchpack.utils.config import configs

class NuScenesTrainer(Trainer):
    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 weight_path: str = None,
                 amp_enabled: bool = False) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.ignore_label = 0

    def _before_train(self) -> None:
        if self.weight_path is not None and os.path.exists(self.weight_path):
            print("load weight from", self.weight_path)
            self.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))
        else:
            print("train from sketch")

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        # num_pts = [coord.size(1) for coord in feed_dict['pixel_coordinates']]

        with amp.autocast(enabled=self.amp_enabled):
            outputs = self.model(in_mod)
            if hasattr(outputs, 'requires_grad'):
                req_grad = outputs.requires_grad
            elif isinstance(outputs, dict):
                t = np.array([getattr(v, 'requires_grad', False) for k, v in outputs.items()])
                req_grad = np.any(t)
            else:
                print("cannot figure out req_grad, default is False")
                req_grad = False
            if req_grad:
                loss_dict = self.criterion(outputs['x_vox'], targets)
        if req_grad:
            predict_vox = loss_dict
            self.summary.add_scalar('ce/vox', predict_vox.item())
            loss = predict_vox
            self.summary.add_scalar('total_loss', loss.item())
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return outputs
        else:
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs_vox = []
            _targets = []
            outputs_vox = outputs.get('x_vox')
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (in_mod['lidar'].C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped_vox = outputs_vox[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs_vox.append(outputs_mapped_vox)
                _targets.append(targets_mapped)
            outputs_vox = torch.cat(_outputs_vox, 0).cpu()
            targets = torch.cat(_targets, 0).cpu()
            return {
                'outputs_vox': outputs_vox,
                'targets': targets,
            }

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass


class NuScenesLCTrainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 weight_path: str = None,
                 pretrain_weight: str = None,
                 amp_enabled: bool = False,) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.pretrain_weight = pretrain_weight
        self.num_classes = self.model.module.num_classes if hasattr(self.model, 'module') else self.model.num_classes
        self.ignore_label = configs['criterion']['ignore_index']

    def _before_train(self) -> None:
        if self.weight_path is not None and os.path.exists(self.weight_path):
            print("load weight from", self.weight_path)
            # state_dict = torch.load(self.weight_path, map_location=torch.device('cpu'))
            # self.model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['model'].items()})
            self.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))
        elif self.pretrain_weight is not None and os.path.exists(self.pretrain_weight):
            print('load pretrained weight from', self.pretrain_weight)
            state_dict = torch.load(self.pretrain_weight, map_location=torch.device('cpu'))
            new_state_dict = {}
            for k, v in state_dict['model'].items():
                if 'classifier' not in k:
                    # new_state_dict[k.replace('module.', '')] = v
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict, strict=False)
        else:
            print("train from sketch")

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        in_mod['images'] = feed_dict['images'].permute(0, 1, 4, 2, 3).contiguous().cuda(non_blocking=True)
        in_mod['pixel_coordinates'] = [coord.cuda(non_blocking=True) for coord in feed_dict['pixel_coordinates']]
        in_mod['masks'] = [mask.cuda(non_blocking=True) for mask in feed_dict['masks']]
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        num_pts = [coord.size(1) for coord in feed_dict['pixel_coordinates']]

        with amp.autocast(enabled=self.amp_enabled):
            outputs = self.model(in_mod)
            if hasattr(outputs, 'requires_grad'):
                req_grad = outputs.requires_grad
            elif isinstance(outputs, dict):
                t = np.array([getattr(v, 'requires_grad', False) for k, v in outputs.items()])
                req_grad = np.any(t)
            else:
                print("cannot figure out req_grad, default is False")
                req_grad = False
            if req_grad:
                loss_dict = self.criterion(outputs, targets)
        if req_grad:
            predict_vox = loss_dict.get('predict_vox')
            predict_pix = loss_dict.get('predict_pix')
            lovasz_vox = loss_dict.get('lovz_vox')
            lovasz_pix = loss_dict.get('lovz_pix')
            self.summary.add_scalar('ce_lovasz/vox', predict_vox.item() + lovasz_vox.item())
            self.summary.add_scalar('ce_lovasz/pix', predict_pix.item() + lovasz_pix.item())
            predict_loss = predict_vox + lovasz_vox + predict_pix + lovasz_pix
            loss = predict_loss
            self.summary.add_scalar('total_loss', loss.item())
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return outputs
        else:
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs_vox, _outputs_pix, _outputs_embed = [], [], []
            _targets = []
            outputs_vox = outputs.get('x_vox')
            outputs_pix = outputs.get('x_pix')
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (in_mod['lidar'].C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped_vox = outputs_vox[cur_scene_pts][cur_inv].argmax(1)
                outputs_mapped_pix = outputs_pix[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs_vox.append(outputs_mapped_vox)
                _outputs_pix.append(outputs_mapped_pix)
                _targets.append(targets_mapped)
            outputs_vox = torch.cat(_outputs_vox, 0).cpu()
            outputs_pix = torch.cat(_outputs_pix, 0).cpu()
            targets = torch.cat(_targets, 0).cpu()
            return {
                'outputs_vox': outputs_vox,
                'outputs_pix': outputs_pix,
                'targets': targets,
            }

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass


class NuScenesLCSparsePretrainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 weight_path: str = None,
                 amp_enabled: bool = False,
                 mm: float = 0.9) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.mm = mm
        self.num_classes = configs['data']['num_classes']
        self.out_channel = self.model.module.out_channel if hasattr(self.model, 'module') else self.model.out_channel
        self.proj_channel = self.model.module.proj_channel if hasattr(self.model, 'module') else self.model.proj_channel
        self.ignore_label = 0
    @torch.no_grad()
    def _calculate_class_anchor(self, embeddings: torch.Tensor, targets: torch.Tensor):

        valid_mask = (targets != self.ignore_label)
        embeddings = embeddings[valid_mask, :]
        targets = targets[valid_mask]

        embeddings = embeddings.view(-1, embeddings.shape[-1])

        prototypes = torch.zeros((self.num_classes, embeddings.shape[-1]),
                                 dtype=embeddings.dtype,
                                 device=embeddings.device)
        u_l, count = torch.unique(targets, return_counts=True)
        exist_mask = torch.zeros(size=(self.num_classes,), device=embeddings.device).bool()
        for i in u_l:
            exist_mask[i] = True
        targets = targets.view(-1, 1).expand(-1, embeddings.shape[-1])
        prototypes.scatter_add_(0, targets, embeddings)
        prototypes[exist_mask, :] = prototypes[exist_mask, :] / count.view(-1, 1)
        return prototypes, exist_mask

    def _before_train(self) -> None:
        if self.weight_path is not None and os.path.exists(self.weight_path):
            print("load weight from", self.weight_path)
            self.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))
        else:
            print("train from sketch")

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        in_mod['images'] = feed_dict['images'].permute(0, 1, 4, 2, 3).contiguous().cuda()
        in_mod['pixel_coordinates'] = [coord.cuda(non_blocking=True) for coord in feed_dict['pixel_coordinates']]
        in_mod['masks'] = [mask.cuda(non_blocking=True) for mask in feed_dict['masks']]
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        sparse_label = feed_dict.get('sparse_label', None)
        prop_label = feed_dict.get('prop_label', None)
        neg_label = feed_dict.get('neg_label', None)
        sparse_label = sparse_label.F.long().cuda(non_blocking=True) if sparse_label is not None else None
        prop_label = prop_label.F.long().cuda(non_blocking=True) if prop_label is not None else None
        neg_label = neg_label.F.long().cuda(non_blocking=True) if neg_label is not None else None

        with amp.autocast(enabled=self.amp_enabled):
            outputs = self.model(in_mod)
            if self.model.training:
                target_dict = {
                    'sparse_label': sparse_label,
                    'prop_label': prop_label,
                    'neg_label': neg_label
                }
                loss_dict = self.criterion(outputs, target_dict)
        if self.model.training:
            sp_vox = loss_dict.get('sp_vox')
            sp_pix = loss_dict.get('sp_pix')
            pp_vox = loss_dict.get('pp_vox')
            pp_pix = loss_dict.get('pp_pix')
            neg_vox = loss_dict.get('neg_vox')
            neg_pix = loss_dict.get('neg_pix')
            self.summary.add_scalar('sp_pp/vox', sp_vox.item() + pp_vox.item())
            self.summary.add_scalar('sp_pp/pix', sp_pix.item() + pp_pix.item())
            self.summary.add_scalar('neg/vox', neg_vox.item())
            self.summary.add_scalar('neg/pix', neg_pix.item())

            predict_loss = sp_vox + pp_vox + sp_pix + pp_pix

            negative_loss = neg_vox + neg_pix

            loss = predict_loss + negative_loss

            self.summary.add_scalar('total_loss', loss.item())
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return outputs
        else:
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs_vox, _outputs_pix = [], []
            _targets = []
            outputs_vox = outputs.get('x_vox')
            outputs_pix = outputs.get('x_pix')
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (in_mod['lidar'].C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped_vox = outputs_vox[cur_scene_pts][cur_inv].argmax(1)
                outputs_mapped_pix = outputs_pix[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs_vox.append(outputs_mapped_vox)
                _outputs_pix.append(outputs_mapped_pix)
                _targets.append(targets_mapped)
            outputs_vox = torch.cat(_outputs_vox, 0).cpu()
            outputs_pix = torch.cat(_outputs_pix, 0).cpu()
            targets = torch.cat(_targets, 0).cpu()
            return {
                'outputs_vox': outputs_vox,
                'outputs_pix': outputs_pix,
                'targets': targets,
            }

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()

        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass


class NuScenesLCSparseAssoP2PPretrainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 weight_path: str = None,
                 amp_enabled: bool = False,
                 mm: float = 0.9) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.mm = mm
        self.num_classes = self.model.module.num_classes if hasattr(self.model, 'module') else self.model.num_classes
        self.out_channel = self.model.module.out_channel if hasattr(self.model, 'module') else self.model.out_channel
        self.proj_channel = self.model.module.proj_channel if hasattr(self.model, 'module') else self.model.proj_channel
        self.ignore_label = 0

    def _before_train(self) -> None:
        if self.weight_path is not None and os.path.exists(self.weight_path):
            print("load weight from", self.weight_path)
            self.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))
        else:
            print("train from sketch")

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        in_mod['images'] = feed_dict['images'].permute(0, 1, 4, 2, 3).contiguous().cuda()
        in_mod['pixel_coordinates'] = [coord.cuda(non_blocking=True) for coord in feed_dict['pixel_coordinates']]
        in_mod['masks'] = [mask.cuda(non_blocking=True) for mask in feed_dict['masks']]
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        sparse_label = feed_dict.get('sparse_label', None)
        prop_label = feed_dict.get('prop_label', None)
        neg_label = feed_dict.get('neg_label', None)
        sparse_label = sparse_label.F.long().cuda(non_blocking=True) if sparse_label is not None else None
        prop_label = prop_label.F.long().cuda(non_blocking=True) if prop_label is not None else None
        neg_label = neg_label.F.long().cuda(non_blocking=True) if neg_label is not None else None

        with amp.autocast(enabled=self.amp_enabled):
            outputs = self.model(in_mod)
            if hasattr(outputs, 'requires_grad'):
                req_grad = outputs.requires_grad
            elif isinstance(outputs, dict):
                t = np.array([getattr(v, 'requires_grad', False) for k, v in outputs.items()])
                req_grad = np.any(t)
            else:
                print("cannot figure out req_grad, default is False")
                req_grad = False
            if req_grad:
                target_dict = {
                    'sparse_label': sparse_label,
                    'prop_label': prop_label,
                    'neg_label': neg_label
                }
                outputs['sp_labels'] = [sp.cuda(non_blocking=True) for sp in feed_dict['sp_labels']]
                outputs['asso_target'] = [sl.cuda(non_blocking=True) for sl in feed_dict['asso_target']]
                loss_dict = self.criterion(outputs, target_dict)
        if req_grad:
            sp_vox = loss_dict.get('sp_vox')
            sp_pix = loss_dict.get('sp_pix')
            pp_vox = loss_dict.get('pp_vox')
            pp_pix = loss_dict.get('pp_pix')
            neg_vox = loss_dict.get('neg_vox')
            neg_pix = loss_dict.get('neg_pix')
            walker_loss = loss_dict.get('asso_loss')
            visit_loss = loss_dict.get('vis_loss')
            # inner_walker_loss = loss_dict.get('inner_asso')
            # inner_visit_loss = loss_dict.get('inner_vis')
            self.summary.add_scalar('sp_pp/vox', sp_vox.item() + pp_vox.item())
            self.summary.add_scalar('sp_pp/pix', sp_pix.item() + pp_pix.item())
            self.summary.add_scalar('neg/vox', neg_vox.item())
            self.summary.add_scalar('neg/pix', neg_pix.item())

            self.summary.add_scalar('asso/walker', walker_loss.item())
            self.summary.add_scalar('asso/vis', visit_loss.item())
            # self.summary.add_scalar('asso/in_walker', inner_walker_loss.item())
            # self.summary.add_scalar('asso/in_vis', inner_visit_loss.item())


            predict_loss = sp_vox + pp_vox + sp_pix + pp_pix
            # predict_loss = sp_vox + sp_pix
            negative_loss = neg_vox + neg_pix
            # asso_loss = walker_loss + 0.5 * visit_loss  # 0.1
            asso_loss = walker_loss + 0.5 * visit_loss
            # inner_asso_loss = inner_walker_loss + 0.5 * inner_visit_loss

            loss = predict_loss + negative_loss + 0.5 * asso_loss
            # loss = predict_loss + negative_loss + 0.5 * asso_loss + 0.5 * inner_asso_loss

            self.summary.add_scalar('total_loss', loss.item())
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return outputs
        else:
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs_vox, _outputs_pix = [], []
            _targets = []
            outputs_vox = outputs.get('x_vox')
            outputs_pix = outputs.get('x_pix')
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (in_mod['lidar'].C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped_vox = outputs_vox[cur_scene_pts][cur_inv].argmax(1)
                outputs_mapped_pix = outputs_pix[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs_vox.append(outputs_mapped_vox)
                _outputs_pix.append(outputs_mapped_pix)
                _targets.append(targets_mapped)
            outputs_vox = torch.cat(_outputs_vox, 0).cpu()
            outputs_pix = torch.cat(_outputs_pix, 0).cpu()
            targets = torch.cat(_targets, 0).cpu()
            return {
                'outputs_vox': outputs_vox,
                'outputs_pix': outputs_pix,
                'targets': targets,
            }

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()

        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])


    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass


class NuScenesLCEstepRunner(Trainer):
    def __init__(self,
                 model: nn.Module,
                 num_workers: int,
                 seed: int,
                 weight_path: str = None,
                 amp_enabled: bool = False,
                 ignore_label: int = 0) -> None:

        self.model = model
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.ignore_label = ignore_label
        self.foreground_mask = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool).cuda()
        self.num_classes = configs['data']['num_classes']
        self.non_dist = configs['non_dist']
        self.oracle_test = False

    def run_estep(self,
                  dataflow: torch.utils.data.DataLoader,
                  *,
                  num_epochs: int = 1,
                  callbacks: Optional[List[Callback]] = None
                  ) -> None:
        if callbacks is None:
            callbacks = []
        callbacks += [
            ProgressBar(),
        ]
        self.train(dataflow=dataflow,
                   num_epochs=num_epochs,
                   callbacks=callbacks)

    @torch.no_grad()
    def filter_pesudo_label_with_dynamic_threshold(self, fmap: torch.Tensor, threshold: float = 0.2,
                                                   softmax=True) -> torch.Tensor:
        """
        :param fmap: vox <Tensor, [N, C]>; pix <Tensor, [6, C, H, W]>
        :param threshold:
        :return:
        """
        assert fmap.dim() == 2 or fmap.dim() == 4
        if fmap.dim() == 2:
            n, c = fmap.size()
        elif fmap.dim() == 4:
            n, c, h, w = fmap.size()
            fmap = fmap.permute(0, 2, 3, 1).contiguous().view(-1, c)
        logits = F.softmax(fmap, dim=1) if softmax else fmap
        val, label = torch.max(logits, dim=1, keepdim=True)
        uq = torch.unique(label)
        th = torch.Tensor([0.5] * c).to(logits.device)
        for ci in uq:
            th[ci] = torch.maximum(torch.max(val[label == ci]) - threshold, th[ci])
        mask = torch.sum(logits >= th, dim=1, keepdim=True)
        label[mask != 1] = self.ignore_label
        return label

    @torch.no_grad()
    def _calculate_class_anchor(self, embeddings: torch.Tensor, targets: torch.Tensor, num_classes):

        embeddings = embeddings.view(-1, embeddings.shape[-1])

        prototypes = torch.zeros((num_classes, embeddings.shape[-1]),
                                 dtype=embeddings.dtype,
                                 device=embeddings.device)
        u_l, count = torch.unique(targets, return_counts=True)
        exist_mask = torch.zeros(size=(num_classes,), device=embeddings.device).bool()
        for i in u_l:
            exist_mask[i] = True
        targets = targets.view(-1, 1).expand(-1, embeddings.shape[-1])
        prototypes = prototypes.scatter_add_(0, targets, embeddings)
        prototypes[exist_mask, :] = prototypes[exist_mask, :] / count.view(-1, 1)
        prototypes[self.ignore_label, :] = 0
        exist_mask[self.ignore_label] = False
        return prototypes, exist_mask

    def _before_train(self) -> None:
        assert self.weight_path is not None and os.path.exists(self.weight_path)
        print("load weight from", self.weight_path)
        if self.non_dist:
            self.model.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu'))['model'])
        else:
            state_dict = torch.load(self.weight_path, map_location=torch.device('cpu'))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['model'].items()})

    def _before_epoch(self) -> None:
        self.model.eval()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        in_mod['images'] = feed_dict['images'].permute(0, 1, 4, 2, 3).contiguous().cuda(non_blocking=True)
        in_mod['pixel_coordinates'] = [coord.cuda(non_blocking=True) for coord in feed_dict['pixel_coordinates']]
        in_mod['masks'] = [mask.cuda(non_blocking=True) for mask in feed_dict['masks']]
        sparse_label = feed_dict['sparse_label'].F.long().cuda(non_blocking=True)
        prop_label = feed_dict['prop_label'].F.long().cuda(non_blocking=True)
        neg_label = feed_dict['neg_label'].F.long().cuda(non_blocking=True)
        lidar_token = feed_dict.get('lidar_token', None)
        pt_with_img_idx = [torch.from_numpy(p[0]) for p in feed_dict.get('pt_with_img_idx', None)]
        full_labels = [torch.from_numpy(l[0]) for l in feed_dict.get('full_labels')]
        lidar_path_list = [p[0] for p in feed_dict.get('lidar_path')]
        sort_inds_list = [torch.from_numpy(i[0]) for i in feed_dict.get('sort_inds')]
        neg_masks = [m[0] for m in feed_dict.get('neg_mask')]

        target = feed_dict.get('target', None)
        if target is not None:
            target = target.F.long().cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):
            with torch.no_grad():
                outputs_dict = self.model(in_mod)
                num_pts = outputs_dict.get('num_pts')
                embedding_vox = outputs_dict.get('embedding_vox')
                x_vox = outputs_dict.get('x_vox')
                # x_pix = outputs_dict.get('x_pix')
                cur = 0
                embedding_labels, pseudo_labels_vox, pseudo_labels_pix = [], [], []
                for n in num_pts:
                    e = embedding_vox[cur:cur + n, :]
                    sp = sparse_label[cur:cur + n]
                    # ne = neg_label[cur:cur+n, :]
                    prototype, exist_mask = self._calculate_class_anchor(e, sp, num_classes=self.num_classes)
                    label_map = torch.nonzero(exist_mask)
                    prototype_t = prototype.transpose(1, 0).contiguous()
                    embedding_score = torch.matmul(e, prototype_t)
                    # embedding_label = torch.zeros(size=(17,), dtype=torch.long, device=embedding_score.device)
                    embedding_score[:, exist_mask] = F.softmax(embedding_score[:, exist_mask], dim=-1)
                    # embedding_labels.append(torch.argmax(embedding_score, dim=-1).long())
                    embedding_label = self.filter_pesudo_label_with_dynamic_threshold(embedding_score, threshold=0.4,  # 0.4
                                                                                      softmax=False)
                    # embedding_label = self.filter_pesudo_label_with_dynamic_threshold(embedding_score[:, exist_mask], threshold=0.1)
                    # embedding_labels.append(label_map[embedding_label].flatten())
                    embedding_labels.append(embedding_label)
                    p_t = self.filter_pesudo_label_with_dynamic_threshold(x_vox[cur:cur + n, :], threshold=0.1)  # default 0.1
                    pseudo_labels_vox.append(p_t)
                    cur += n
                embedding_labels = torch.cat(embedding_labels, dim=0).view_as(sparse_label)
                pseudo_labels_vox = torch.cat(pseudo_labels_vox, dim=0).view_as(sparse_label)
                foreground = self.foreground_mask[pseudo_labels_vox]


                disagree_mask = (embedding_labels != pseudo_labels_vox) & (
                            embedding_labels != self.ignore_label) & foreground
                pseudo_labels_vox[disagree_mask] = self.ignore_label

                neg_mask = torch.cat(
                    [torch.arange(pseudo_labels_vox.size(0)).view(-1, 1).cuda(), pseudo_labels_vox.view(-1, 1)],
                    dim=-1).long()
                neg_mask = neg_label[neg_mask[:, 0], neg_mask[:, 1]]
                neg_mask = (neg_mask == self.ignore_label)
                pseudo_labels_vox[neg_mask] = self.ignore_label

                valid_mask = (sparse_label != self.ignore_label)
                pseudo_labels_vox[valid_mask] = sparse_label[valid_mask]

                pseudo_label_mix = pseudo_labels_vox

                invs = feed_dict['inverse_map']
                _outputs_vox, _outputs_pix = [], []
                full_labels_list = []
                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (in_mod['lidar'].C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    pseudo_labels_mix_full = torch.full_like(pt_with_img_idx[idx], fill_value=self.ignore_label,
                                                             dtype=torch.long)
                    pseudo_labels_mix_full[pt_with_img_idx[idx]] = torch.scatter(
                        pseudo_labels_mix_full[pt_with_img_idx[idx]], dim=0, index=sort_inds_list[idx],
                        src=pseudo_label_mix[cur_scene_pts][cur_inv].cpu())
                    _outputs_vox.append(pseudo_labels_mix_full[neg_masks[idx]])
                    # _outputs_vox.append(pseudo_labels_mix_full)
                    # pseudo_label_mix_full[pt_with_img_idx[idx]] = pseudo_label_mix[cur_scene_pts][cur_inv].cpu()
                    labels_full = torch.full_like(pt_with_img_idx[idx], fill_value=self.ignore_label, dtype=torch.long)
                    labels_full[pt_with_img_idx[idx]] = torch.scatter(labels_full[pt_with_img_idx[idx]], dim=0,
                                                                      index=sort_inds_list[idx], src=full_labels[idx])
                    # labels_full[pt_with_img_idx[idx]] = full_labels[idx]
                    full_labels_list.append(labels_full[neg_masks[idx]])
                return {
                    'pseudo_label_list': _outputs_vox,
                    'ground_truth_list': full_labels_list,
                    'lidar_token_list': lidar_token,
                    'lidar_path_list': lidar_path_list,
                    'neg_masks': neg_masks
                }

class NuScenesLCEstepRunner2(Trainer):
    def __init__(self,
                 model: nn.Module,
                 num_workers: int,
                 seed: int,
                 weight_path: str = None,
                 amp_enabled: bool = False,
                 ignore_label: int = 0) -> None:

        self.model = model
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.ignore_label = ignore_label
        self.foreground_mask = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                            dtype=torch.bool).cuda()
        self.non_dist = configs['non_dist']
        self.oracle_test = True

    def run_estep(self,
                  dataflow: torch.utils.data.DataLoader,
                  *,
                  num_epochs: int = 1,
                  callbacks: Optional[List[Callback]] = None
                  ) -> None:
        if callbacks is None:
            callbacks = []
        callbacks += [
            # MetaInfoSaver(),
            # ConsoleWriter(),
            # TFEventWriter(),
            # JSONLWriter(),
            ProgressBar(),
            # EstimatedTimeLeft()
        ]
        self.train(dataflow=dataflow,
                   num_epochs=num_epochs,
                   callbacks=callbacks)

    @torch.no_grad()
    def filter_pesudo_label_with_dynamic_threshold(self, fmap: torch.Tensor, threshold: float = 0.2,
                                                   softmax=True) -> torch.Tensor:
        """
        :param fmap: vox <Tensor, [N, C]>; pix <Tensor, [6, C, H, W]>
        :param threshold:
        :return:
        """
        assert fmap.dim() == 2 or fmap.dim() == 4
        if fmap.dim() == 2:
            n, c = fmap.size()
        elif fmap.dim() == 4:
            n, c, h, w = fmap.size()
            fmap = fmap.permute(0, 2, 3, 1).contiguous().view(-1, c)
        logits = F.softmax(fmap, dim=1) if softmax else fmap
        val, label = torch.max(logits, dim=1, keepdim=True)
        uq = torch.unique(label)
        th = torch.Tensor([0.5] * c).to(logits.device)
        for ci in uq:
            th[ci] = torch.maximum(torch.max(val[label == ci]) - threshold, th[ci])
        mask = torch.sum(logits >= th, dim=1, keepdim=True)
        label[mask != 1] = self.ignore_label
        return label

    @torch.no_grad()
    def _calculate_class_anchor(self, embeddings: torch.Tensor, targets: torch.Tensor, num_classes):

        embeddings = embeddings.view(-1, embeddings.shape[-1])

        prototypes = torch.zeros((num_classes, embeddings.shape[-1]),
                                 dtype=embeddings.dtype,
                                 device=embeddings.device)
        u_l, count = torch.unique(targets, return_counts=True)
        exist_mask = torch.zeros(size=(num_classes,), device=embeddings.device).bool()
        for i in u_l:
            exist_mask[i] = True
        targets = targets.view(-1, 1).expand(-1, embeddings.shape[-1])
        prototypes = prototypes.scatter_add_(0, targets, embeddings)
        prototypes[exist_mask, :] = prototypes[exist_mask, :] / count.view(-1, 1)
        prototypes[self.ignore_label, :] = 0
        return prototypes, exist_mask

    def _before_train(self) -> None:

        assert self.weight_path is not None and os.path.exists(self.weight_path)
        print("load weight from", self.weight_path)
        if self.non_dist:
            self.model.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu'))['model'])
        else:
            state_dict = torch.load(self.weight_path, map_location=torch.device('cpu'))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['model'].items()})

    def _before_epoch(self) -> None:
        self.model.eval()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        in_mod['images'] = feed_dict['images'].permute(0, 1, 4, 2, 3).contiguous().cuda(non_blocking=True)
        in_mod['pixel_coordinates'] = [coord.cuda(non_blocking=True) for coord in feed_dict['pixel_coordinates']]
        in_mod['masks'] = [mask.cuda(non_blocking=True) for mask in feed_dict['masks']]
        sparse_label = feed_dict['sparse_label'].F.long().cuda(non_blocking=True)
        # prop_label = feed_dict['prop_label'].F.long().cuda(non_blocking=True)
        # neg_label = feed_dict['neg_label'].F.long().cuda(non_blocking=True)
        pseudo_label = feed_dict['pseudo_label'].F.long().cuda(non_blocking=True)
        lidar_token = feed_dict.get('lidar_token', None)
        pt_with_img_idx = [torch.from_numpy(p[0]) for p in feed_dict.get('pt_with_img_idx', None)]
        full_labels = [torch.from_numpy(l[0]) for l in feed_dict.get('full_labels')]
        lidar_path_list = [p[0] for p in feed_dict.get('lidar_path')]
        sort_inds_list = [torch.from_numpy(i[0]) for i in feed_dict.get('sort_inds')]
        # neg_masks = [m[0] for m in feed_dict.get('neg_mask')]
        other_masks = [m[0] for m in feed_dict.get('other_mask')]

        target = feed_dict.get('target', None)
        if target is not None:
            target = target.F.long().cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):
            with torch.no_grad():
                outputs_dict = self.model(in_mod)
                num_pts = outputs_dict.get('num_pts')
                embedding_vox = outputs_dict.get('embedding_vox')
                x_vox = outputs_dict.get('x_vox')
                cur = 0
                embedding_labels, pseudo_labels_vox, pseudo_labels_pix = [], [], []
                for n in num_pts:
                    e = embedding_vox[cur:cur + n, :]
                    sp = sparse_label[cur:cur + n]
                    prototype, exist_mask = self._calculate_class_anchor(e, sp, num_classes=17)
                    prototype_t = prototype.transpose(1, 0).contiguous()
                    embedding_score = torch.matmul(e, prototype_t)
                    embedding_score[:, exist_mask] = F.softmax(embedding_score[:, exist_mask], dim=-1)
                    embedding_label = self.filter_pesudo_label_with_dynamic_threshold(embedding_score, threshold=0.4,
                                                                                      softmax=False)
                    embedding_labels.append(embedding_label)
                    p_t = self.filter_pesudo_label_with_dynamic_threshold(x_vox[cur:cur + n, :], threshold=0.1)  # default 0.1
                    pseudo_labels_vox.append(p_t)
                    cur += n
                embedding_labels = torch.cat(embedding_labels, dim=0).view_as(sparse_label)
                pseudo_labels_vox = torch.cat(pseudo_labels_vox, dim=0).view_as(sparse_label)
                foremask = self.foreground_mask[pseudo_labels_vox]

                disagree_mask = (embedding_labels != pseudo_labels_vox) & (embedding_labels != self.ignore_label) & foremask
                pseudo_labels_vox[disagree_mask] = self.ignore_label

                neg_mask = pseudo_label != self.ignore_label
                pseudo_labels_vox[neg_mask] = pseudo_label[neg_mask]

                valid_mask = (sparse_label != self.ignore_label)
                pseudo_labels_vox[valid_mask] = sparse_label[valid_mask]

                pseudo_label_mix = pseudo_labels_vox


                invs = feed_dict['inverse_map']
                _outputs_vox, _outputs_pix = [], []
                full_labels_list = []
                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (in_mod['lidar'].C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    pseudo_labels_mix_full = torch.full_like(pt_with_img_idx[idx], fill_value=self.ignore_label,
                                                             dtype=torch.long)
                    pseudo_labels_mix_full[pt_with_img_idx[idx]] = torch.scatter(
                        pseudo_labels_mix_full[pt_with_img_idx[idx]], dim=0, index=sort_inds_list[idx],
                        src=pseudo_label_mix[cur_scene_pts][cur_inv].cpu())
                    _outputs_vox.append(pseudo_labels_mix_full[other_masks[idx]])
                    labels_full = torch.full_like(pt_with_img_idx[idx], fill_value=self.ignore_label, dtype=torch.long)
                    labels_full[pt_with_img_idx[idx]] = torch.scatter(labels_full[pt_with_img_idx[idx]], dim=0,
                                                                      index=sort_inds_list[idx], src=full_labels[idx])
                    full_labels_list.append(labels_full[other_masks[idx]])
                return {
                    'pseudo_label_list': _outputs_vox,
                    'ground_truth_list': full_labels_list,
                    'lidar_token_list': lidar_token,
                    'lidar_path_list': lidar_path_list,
                    'neg_masks': other_masks
                }


class NuScenesLCSparseMStepRunner(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 weight_path: str = None,
                 amp_enabled: bool = False,
                 mm: float = 0.9) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.mm = mm
        self.num_classes = self.model.module.num_classes if hasattr(self.model, 'module') else self.model.num_classes
        self.out_channel = self.model.module.out_channel if hasattr(self.model, 'module') else self.model.out_channel
        self.proj_channel = self.model.module.proj_channel if hasattr(self.model, 'module') else self.model.proj_channel
        self.ignore_label = 0

    def _before_train(self) -> None:
        if self.weight_path is not None and os.path.exists(self.weight_path):
            print("load weight from", self.weight_path)
            self.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))
        else:
            print("train from sketch")

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        in_mod['images'] = feed_dict['images'].permute(0, 1, 4, 2, 3).contiguous().cuda()
        in_mod['pixel_coordinates'] = [coord.cuda(non_blocking=True) for coord in feed_dict['pixel_coordinates']]
        in_mod['masks'] = [mask.cuda(non_blocking=True) for mask in feed_dict['masks']]
        # targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        sparse_label = feed_dict.get('sparse_label', None)
        prop_label = feed_dict.get('prop_label', None)
        neg_label = feed_dict.get('neg_label', None)
        pseudo_label = feed_dict.get('pseudo_label', None)
        sparse_label = sparse_label.F.long().cuda(non_blocking=True) if sparse_label is not None else None
        prop_label = prop_label.F.long().cuda(non_blocking=True) if prop_label is not None else None
        neg_label = neg_label.F.long().cuda(non_blocking=True) if neg_label is not None else None
        pseudo_label = pseudo_label.F.long().cuda(non_blocking=True) if pseudo_label is not None else None

        with amp.autocast(enabled=self.amp_enabled):
            outputs = self.model(in_mod)
            if hasattr(outputs, 'requires_grad'):
                req_grad = outputs.requires_grad
            elif isinstance(outputs, dict):
                t = np.array([getattr(v, 'requires_grad', False) for k, v in outputs.items()])
                req_grad = np.any(t)
            else:
                print("cannot figure out req_grad, default is False")
                req_grad = False
            if req_grad:
                target_dict = {
                    'sparse_label': sparse_label,
                    'prop_label': prop_label,
                    'neg_label': neg_label,
                    'pseudo_label': pseudo_label
                }
                loss_dict = self.criterion(outputs, target_dict)
        if req_grad:
            sp_vox = loss_dict.get('sp_vox')
            sp_pix = loss_dict.get('sp_pix')
            pp_vox = loss_dict.get('pp_vox')
            pp_pix = loss_dict.get('pp_pix')
            neg_vox = loss_dict.get('neg_vox')
            neg_pix = loss_dict.get('neg_pix')
            pl_vox = loss_dict.get('pl_vox')
            pl_pix = loss_dict.get('pl_pix')
            self.summary.add_scalar('sp_pp/vox', sp_vox.item() + pp_vox.item())
            self.summary.add_scalar('sp_pp/pix', sp_pix.item() + pp_pix.item())
            self.summary.add_scalar('neg/vox', neg_vox.item())
            self.summary.add_scalar('neg/pix', neg_pix.item())
            self.summary.add_scalar('pl/vox', pl_vox.item())
            self.summary.add_scalar('pl/pix', pl_pix.item())

            predict_loss = (sp_vox + pp_vox) / 2 + (sp_pix + pp_pix) / 2
            negative_loss = (neg_vox + neg_pix) / 2
            pl_loss = (pl_vox + pl_pix) / 2
            loss = predict_loss + negative_loss + pl_loss

            self.summary.add_scalar('total_loss', loss.item())
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return outputs
        else:
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs_vox, _outputs_pix = [], []
            _targets = []
            outputs_vox = outputs.get('x_vox')
            outputs_pix = outputs.get('x_pix')
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (in_mod['lidar'].C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped_vox = outputs_vox[cur_scene_pts][cur_inv].argmax(1)
                outputs_mapped_pix = outputs_pix[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs_vox.append(outputs_mapped_vox)
                _outputs_pix.append(outputs_mapped_pix)
                _targets.append(targets_mapped)
            outputs_vox = torch.cat(_outputs_vox, 0).cpu()
            outputs_pix = torch.cat(_outputs_pix, 0).cpu()
            targets = torch.cat(_targets, 0).cpu()
            return {
                'outputs_vox': outputs_vox,
                'outputs_pix': outputs_pix,
                'targets': targets,
            }

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass


class NuScenesLCSparseAssoP2PMStepRunner(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 weight_path: str = None,
                 amp_enabled: bool = False,
                 mm: float = 0.9) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.mm = mm
        self.num_classes = self.model.module.num_classes if hasattr(self.model, 'module') else self.model.num_classes
        self.out_channel = self.model.module.out_channel if hasattr(self.model, 'module') else self.model.out_channel
        self.proj_channel = self.model.module.proj_channel if hasattr(self.model, 'module') else self.model.proj_channel
        self.ignore_label = 0

    def _before_train(self) -> None:
        if self.weight_path is not None and os.path.exists(self.weight_path):
            print("load weight from", self.weight_path)
            self.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))
        else:
            print("train from sketch")

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        in_mod['images'] = feed_dict['images'].permute(0, 1, 4, 2, 3).contiguous().cuda(non_blocking=True)
        in_mod['pixel_coordinates'] = [coord.cuda(non_blocking=True) for coord in feed_dict['pixel_coordinates']]
        in_mod['masks'] = [mask.cuda(non_blocking=True) for mask in feed_dict['masks']]
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        sparse_label = feed_dict.get('sparse_label', None)
        prop_label = feed_dict.get('prop_label', None)
        neg_label = feed_dict.get('neg_label', None)
        pseudo_label = feed_dict.get('pseudo_label', None)
        sparse_label = sparse_label.F.long().cuda(non_blocking=True) if sparse_label is not None else None
        prop_label = prop_label.F.long().cuda(non_blocking=True) if prop_label is not None else None
        neg_label = neg_label.F.long().cuda(non_blocking=True) if neg_label is not None else None
        pseudo_label = pseudo_label.F.long().cuda(non_blocking=True) if pseudo_label is not None else None

        with amp.autocast(enabled=self.amp_enabled):
            outputs = self.model(in_mod)
            if hasattr(outputs, 'requires_grad'):
                req_grad = outputs.requires_grad
            elif isinstance(outputs, dict):
                t = np.array([getattr(v, 'requires_grad', False) for k, v in outputs.items()])
                req_grad = np.any(t)
            else:
                print("cannot figure out req_grad, default is False")
                req_grad = False
            if req_grad:
                target_dict = {
                    'sparse_label': sparse_label,
                    'prop_label': prop_label,
                    'neg_label': neg_label,
                    'pseudo_label': pseudo_label
                }
                outputs['sp_labels'] = [sp.cuda(non_blocking=True) for sp in feed_dict['sp_labels']]
                outputs['asso_target'] = [sl.cuda(non_blocking=True) for sl in feed_dict['asso_target']]
                loss_dict = self.criterion(outputs, target_dict)
        if req_grad:
            sp_vox = loss_dict.get('sp_vox')
            sp_pix = loss_dict.get('sp_pix')
            pp_vox = loss_dict.get('pp_vox')
            pp_pix = loss_dict.get('pp_pix')
            neg_vox = loss_dict.get('neg_vox')
            neg_pix = loss_dict.get('neg_pix')

            walker_loss = loss_dict.get('asso_loss')
            visit_loss = loss_dict.get('vis_loss')
            pl_vox = loss_dict.get('pl_vox')
            pl_pix = loss_dict.get('pl_pix')

            self.summary.add_scalar('sp_pp/vox', sp_vox.item() + pp_vox.item())
            self.summary.add_scalar('sp_pp/pix', sp_pix.item() + pp_pix.item())
            self.summary.add_scalar('neg/vox', neg_vox.item())
            self.summary.add_scalar('neg/pix', neg_pix.item())
            self.summary.add_scalar('asso/walker', walker_loss.item())
            self.summary.add_scalar('asso/vis', visit_loss.item())
            self.summary.add_scalar('pl/vox', pl_vox.item())
            self.summary.add_scalar('pl/pix', pl_pix.item())

            predict_loss = sp_vox + pp_vox + sp_pix + pp_pix
            negative_loss = neg_vox + neg_pix
            pl_loss = pl_vox + pl_pix
            asso_loss = walker_loss + 0.5 * visit_loss
            loss = predict_loss + negative_loss + pl_loss + 0.5 * asso_loss

            self.summary.add_scalar('total_loss', loss.item())
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return outputs
        else:
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs_vox, _outputs_pix = [], []
            _targets = []
            outputs_vox = outputs.get('x_vox')
            outputs_pix = outputs.get('x_pix')
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (in_mod['lidar'].C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped_vox = outputs_vox[cur_scene_pts][cur_inv].argmax(1)
                outputs_mapped_pix = outputs_pix[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs_vox.append(outputs_mapped_vox)
                _outputs_pix.append(outputs_mapped_pix)
                _targets.append(targets_mapped)
            outputs_vox = torch.cat(_outputs_vox, 0).cpu()
            outputs_pix = torch.cat(_outputs_pix, 0).cpu()
            targets = torch.cat(_targets, 0).cpu()
            return {
                'outputs_vox': outputs_vox,
                'outputs_pix': outputs_pix,
                'targets': targets,
            }

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass

class NuScenes_EM_MIOURunner(Trainer):
    def __init__(self,
                 model: nn.Module,
                 num_workers: int,
                 seed: int,
                 weight_path: str = None,
                 amp_enabled: bool = False,
                 ignore_label: int = 0) -> None:

        self.model = model
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.ignore_label = ignore_label
        self.num_total = 0
        self.num_sub = 0

    def run_miou(self,
                 dataflow: torch.utils.data.DataLoader,
                 *,
                 num_epochs: int = 1,
                 callbacks: Optional[List[Callback]] = None
                 ) -> None:
        if callbacks is None:
            callbacks = []
        callbacks += [
            ProgressBar(),
        ]
        self.train(dataflow=dataflow,
                   num_epochs=num_epochs,
                   callbacks=callbacks)

    def _before_train(self) -> None:

        assert self.weight_path is not None and os.path.exists(self.weight_path)
        print("load weight from", self.weight_path)
        state_dict = torch.load(self.weight_path, map_location=torch.device('cpu'))
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['model'].items()})
        # self.model.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu'))['model'])

    def _before_epoch(self) -> None:
        self.model.eval()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        in_mod['images'] = feed_dict['images'].permute(0, 1, 4, 2, 3).contiguous().cuda(non_blocking=True)
        in_mod['pixel_coordinates'] = [coord.cuda() for coord in feed_dict['pixel_coordinates']]
        in_mod['masks'] = [mask.cuda() for mask in feed_dict['masks']]
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        lidar_token = feed_dict['lidar_token']
        # view_mask = [m.cuda() for m in feed_dict['pt_with_img_idx']]
        # num_pts = [coord.size(1) for coord in feed_dict['pixel_coordinates']]

        # self.num_total += np.sum(np.array(feed_dict['num_total']))
        # self.num_sub += np.sum(np.array(feed_dict['num_sub']))
        #
        # print('sub rate:', float(self.num_sub) / self.num_total)

        with amp.autocast(enabled=self.amp_enabled):
            with torch.no_grad():
                outputs = self.model(in_mod)
                invs = feed_dict['inverse_map']
                all_labels = feed_dict['targets_mapped']
                _outputs_vox, _outputs_pix, _outputs_embed = [], [], []
                _targets = []
                outputs_vox = outputs.get('x_vox')
                # outputs_vox = outputs
                num_pts = []
                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (in_mod['lidar'].C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                    outputs_mapped_vox = outputs_vox[cur_scene_pts][cur_inv].argmax(1)
                    num_pts.append(outputs_mapped_vox.size(0))
                    targets_mapped = all_labels.F[cur_label]
                    _outputs_vox.append(outputs_mapped_vox)
                    _targets.append(targets_mapped)
                outputs_vox = torch.cat(_outputs_vox, 0).cpu()
                targets = torch.cat(_targets, 0).cpu()
                return {
                    'outputs_vox': outputs_vox,
                    'targets': targets,
                    'lidar_token_list': lidar_token,
                    'num_pts': num_pts
                }
