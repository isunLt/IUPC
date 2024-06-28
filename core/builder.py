from typing import Callable

import torch
import torch.optim
from torch import nn
import torchpack.distributed as dist
from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler

__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
    'make_scheduler'
]


def make_dataset(dataset_name: str = None, **kwargs) -> dict:
    if dataset_name is None:
        dataset_name = configs.dataset.name
    if dataset_name == 'semantic_kitti':
        from core.datasets.semantic_kitti import SemanticKITTI
        dataset = SemanticKITTI(root=configs.dataset.root,
                                voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'lc_semantic_kitti':
    #     from core.datasets import LCSemanticKITTI
    #     dataset = LCSemanticKITTI(root=configs.dataset.root,
    #                               voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'sparse_lc_semantic_kitti':
    #     from core.datasets.sparse_lc_semantic_kitti import SparseLCSemanticKITTI
    #     dataset = SparseLCSemanticKITTI(root=configs.dataset.root,
    #                                     voxel_size=configs.dataset.voxel_size,
    #                                     sparse_label_root=configs.dataset.sparse_label_root)
    # elif dataset_name == 'semantic_kitti_preseg':
    #     from core.datasets.kitti_presegmentation import SemKITTISparseLabelPrepare
    #     dataset = SemKITTISparseLabelPrepare(root=configs.dataset.root,
    #                                          voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'semantic_kitti_superpixel':
    #     from core.datasets.prepare_superpixel_kitti import SuperpixelSemanticKITTI
    #     dataset = SuperpixelSemanticKITTI(root=configs.dataset.root,
    #                                       voxel_size=configs.dataset.voxel_size,
    #                                       superpixel=configs.dataset.superpixel,
    #                                       sparse_label_root=configs.dataset.sparse_label_root)
    # elif dataset_name == 'semantic_kitti_assop2p':
    #     from core.datasets.sparse_pretrain_assop2p_lc_kitti import SparseAssoP2PLCSemanticKITTI
    #     dataset = SparseAssoP2PLCSemanticKITTI(root=configs.dataset.root,
    #                                            voxel_size=configs.dataset.voxel_size,
    #                                            superpixel=configs.dataset.superpixel,
    #                                            sparse_label_root=configs.dataset.sparse_label_root)
    # elif dataset_name == 'sparse_estep_lc_kitti':
    #     from core.datasets.sparse_estep_lc_kitti import SparseEStepLCKitti
    #     dataset = SparseEStepLCKitti(
    #         root=configs.dataset.root,
    #         voxel_size=configs.dataset.voxel_size,
    #         sparse_label_root=configs.dataset.sparse_label_root
    #     )
    # elif dataset_name == 'sparse_estep_lc_kitti_2':
    #     from core.datasets.sparse_estep_lc_kitti_2 import SparseEStepLCKitti2
    #     pseudo_label_dir = kwargs.get('pseudo_label_dir')
    #     dataset = SparseEStepLCKitti2(
    #         root=configs.dataset.root,
    #         voxel_size=configs.dataset.voxel_size,
    #         sparse_label_root=configs.dataset.sparse_label_root,
    #         pseudo_label_dir=pseudo_label_dir
    #     )
    # elif dataset_name == 'sparse_mstep_assop2p_lc_kitti':
    #     from core.datasets.sparse_mstep_assop2p_lc_kitti import SparseAssoP2PMStepLCSemanticKITTI
    #     pseudo_label_dir = kwargs.get('pseudo_label_dir', None)
    #     dataset = SparseAssoP2PMStepLCSemanticKITTI(
    #         root=configs.dataset.root,
    #         voxel_size=configs.dataset.voxel_size,
    #         superpixel=configs.dataset.superpixel,
    #         sparse_label_root=configs.dataset.sparse_label_root,
    #         pseudo_label_dir=pseudo_label_dir
    #     )
    elif dataset_name == "semantic_nusc":
        from core.datasets import NuScenes
        dataset = NuScenes(root=configs.dataset.root,
                           voxel_size=configs.dataset.voxel_size,
                           version="v1.0-trainval",
                           verbose=True)
    elif dataset_name == "lc_semantic_nusc":
        from core.datasets.lc_semantic_nusc import LCNuScenes
        dataset = LCNuScenes(root=configs.dataset.root,
                             voxel_size=configs.dataset.voxel_size,
                             version=configs.dataset.version,
                             verbose=True,
                             image_crop_rate=configs.dataset.image_crop_rate)
    elif dataset_name == 'random_label_lc_nusc':
        from core.datasets.random_pretrain_lc_nusc import NuScenesLCRandomLabel
        dataset = NuScenesLCRandomLabel(
            root=configs.dataset.root,
            image_crop_rate=configs.dataset.image_crop_rate,
            voxel_size=configs.dataset.voxel_size,
            version=configs.dataset.version,
            use_color=configs.dataset.use_color,
            use_augment=configs.dataset.use_augment,
            pts_sr=configs.dataset.pts_sr,
            verbose=True
        )
    elif dataset_name == 'prepare_sparse_label':
        from core.datasets.prepare_sparse_label_mask import NuScenesSparseLabelPrepare
        dataset = NuScenesSparseLabelPrepare(
            root=configs.dataset.root,
            version=configs.dataset.version,
            verbose=True
        )
    elif dataset_name == 'sparse_label_lc_nusc':
        from core.datasets import NuScenesLCSparseLabel
        dataset = NuScenesLCSparseLabel(
            root=configs.dataset.root,
            image_crop_rate=configs.dataset.image_crop_rate,
            voxel_size=configs.dataset.voxel_size,
            version=configs.dataset.version,
            use_color=configs.dataset.use_color,
            use_augment=configs.dataset.use_augment,
            verbose=True
        )
    elif dataset_name == 'sparse_estep_lc_nusc':
        from core.datasets import SparseEStepLCNusc
        dataset = SparseEStepLCNusc(
            root=configs.dataset.root,
            image_crop_rate=configs.dataset.image_crop_rate,
            voxel_size=configs.dataset.voxel_size,
            version=configs.dataset.version,
            use_color=configs.dataset.use_color,
            use_augment=configs.dataset.use_augment,
            verbose=True
        )
    elif dataset_name == 'sparse_estep_lc_nusc_2':
        from core.datasets.sparse_estep_lc_nusc_2 import SparseEStepLCNusc2
        pseudo_label_dir = kwargs.get('pseudo_label_dir', None)
        dataset = SparseEStepLCNusc2(
            root=configs.dataset.root,
            image_crop_rate=configs.dataset.image_crop_rate,
            voxel_size=configs.dataset.voxel_size,
            version=configs.dataset.version,
            use_color=configs.dataset.use_color,
            use_augment=configs.dataset.use_augment,
            pseudo_label_dir=pseudo_label_dir,
            verbose=True
        )
    elif dataset_name == 'sparse_estep_lc_nusc_3':
        from core.datasets.sparse_estep_lc_nusc_3 import SparseEStepLCNusc3
        pseudo_label_dir = kwargs.get('pseudo_label_dir', None)
        dataset = SparseEStepLCNusc3(
            root=configs.dataset.root,
            image_crop_rate=configs.dataset.image_crop_rate,
            voxel_size=configs.dataset.voxel_size,
            version=configs.dataset.version,
            use_color=configs.dataset.use_color,
            use_augment=configs.dataset.use_augment,
            pseudo_label_dir=pseudo_label_dir,
            verbose=True
        )
    elif dataset_name == 'sparse_label_mstep_lc_nusc':
        from core.datasets.sparse_mstep_lc_nusc import NuScenesLCPLSparseLabel
        pseudo_label_dir = kwargs.get('pseudo_label_dir', None)
        dataset = NuScenesLCPLSparseLabel(
            root=configs.dataset.root,
            image_crop_rate=configs.dataset.image_crop_rate,
            voxel_size=configs.dataset.voxel_size,
            version=configs.dataset.version,
            use_color=configs.dataset.use_color,
            use_augment=configs.dataset.use_augment,
            pseudo_label_dir=pseudo_label_dir,
            verbose=True
        )
    elif dataset_name == 'sparse_label_mstep_assop2p_lc_nusc':
        from core.datasets.sparse_mstep_assop2p_lc_nusc import NuScenesAssoP2PLCPLSparseLabel
        pseudo_label_dir = kwargs.get('pseudo_label_dir', None)
        dataset = NuScenesAssoP2PLCPLSparseLabel(
            root=configs.dataset.root,
            image_crop_rate=configs.dataset.image_crop_rate,
            voxel_size=configs.dataset.voxel_size,
            version=configs.dataset.version,
            use_color=configs.dataset.use_color,
            use_augment=configs.dataset.use_augment,
            pseudo_label_dir=pseudo_label_dir,
            superpixel=configs.dataset.superpixel,
            verbose=True
        )
    elif dataset_name == 'sparse_label_mstep_assop2p_lc_nusc_2':
        from core.datasets.sparse_mstep_assop2p_lc_nusc_2 import NuScenesAssoP2PLCPLSparseLabel2
        pseudo_label_dir = kwargs.get('pseudo_label_dir', None)
        dataset = NuScenesAssoP2PLCPLSparseLabel2(
            root=configs.dataset.root,
            image_crop_rate=configs.dataset.image_crop_rate,
            voxel_size=configs.dataset.voxel_size,
            version=configs.dataset.version,
            use_color=configs.dataset.use_color,
            use_augment=configs.dataset.use_augment,
            pseudo_label_dir=pseudo_label_dir,
            superpixel=configs.dataset.superpixel,
            verbose=True
        )
    elif dataset_name == 'sparse_assop2p_lc_nusc':
        from core.datasets.sparse_pretrain_assop2p_lc_nusc import NuScenesLCAssoP2PSparseLabel
        dataset = NuScenesLCAssoP2PSparseLabel(
            root=configs.dataset.root,
            image_crop_rate=configs.dataset.image_crop_rate,
            voxel_size=configs.dataset.voxel_size,
            version=configs.dataset.version,
            use_color=configs.dataset.use_color,
            use_augment=configs.dataset.use_augment,
            superpixel=configs.dataset.superpixel,
            verbose=True
        )
    elif dataset_name == 'lc_semantic_nusc_eval':
        from core.datasets.lc_semantic_nusc_eval import LCNuScenesEval
        dataset = LCNuScenesEval(
            root=configs.dataset.root,
            image_crop_rate=configs.dataset.image_crop_rate,
            voxel_size=configs.dataset.voxel_size,
            version=configs.dataset.version,
            use_color=configs.dataset.use_color,
            use_augment=configs.dataset.use_augment,
            verbose=True
        )
    # elif dataset_name == 'semantic_waymo':
    #     from core.datasets.semantic_waymo import SemanticWaymo
    #     dataset = SemanticWaymo(root=configs.dataset.root,
    #                             voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'lc_semantic_waymo':
    #     from core.datasets.lc_semantic_waymo import LCSemanticWaymo
    #     dataset = LCSemanticWaymo(root=configs.dataset.root,
    #                               voxel_size=configs.dataset.voxel_size,
    #                               image_crop_rate=configs.dataset.image_crop_rate)
    # elif dataset_name == 'sparse_lc_waymo':
    #     from core.datasets.sparse_pretrain_lc_waymo import SparseLCWaymo
    #     dataset = SparseLCWaymo(root=configs.dataset.root,
    #                             voxel_size=configs.dataset.voxel_size,
    #                             image_crop_rate=configs.dataset.image_crop_rate,
    #                             sparse_label_root=configs.dataset.sparse_label_root)
    # elif dataset_name == 'sparse_assop2p_lc_waymo':
    #     from core.datasets.sparse_pretrain_assop2p_lc_waymo import SparseAssop2pLCWaymo
    #     dataset = SparseAssop2pLCWaymo(root=configs.dataset.root,
    #                                    voxel_size=configs.dataset.voxel_size,
    #                                    image_crop_rate=configs.dataset.image_crop_rate,
    #                                    sparse_label_root=configs.dataset.sparse_label_root,
    #                                    superpixel=configs.dataset.superpixel)
    # elif dataset_name == 'sparse_estep_lc_waymo':
    #     from core.datasets.sparse_estep_lc_waymo import SparseEStepLCWaymo
    #     dataset = SparseEStepLCWaymo(root=configs.dataset.root,
    #                                  voxel_size=configs.dataset.voxel_size,
    #                                  image_crop_rate=configs.dataset.image_crop_rate,
    #                                  sparse_label_root=configs.dataset.sparse_label_root)
    # elif dataset_name == 'sparse_estep_lc_waymo_2':
    #     from core.datasets.sparse_estep_lc_waymo_2 import SparseEStepLCWaymo2
    #     pseudo_label_dir = kwargs.get('pseudo_label_dir', None)
    #     dataset = SparseEStepLCWaymo2(root=configs.dataset.root,
    #                                   voxel_size=configs.dataset.voxel_size,
    #                                   image_crop_rate=configs.dataset.image_crop_rate,
    #                                   sparse_label_root=configs.dataset.sparse_label_root,
    #                                   pseudo_label_dir=pseudo_label_dir)
    # elif dataset_name == 'sparse_mstep_assop2p_lc_waymo':
    #     from core.datasets.sparse_mstep_assop2p_lc_waymo import SparseMStepAssop2pLCWaymo
    #     pseudo_label_dir = kwargs.get('pseudo_label_dir')
    #     dataset = SparseMStepAssop2pLCWaymo(root=configs.dataset.root,
    #                                         voxel_size=configs.dataset.voxel_size,
    #                                         image_crop_rate=configs.dataset.image_crop_rate,
    #                                         sparse_label_root=configs.dataset.sparse_label_root,
    #                                         superpixel=configs.dataset.superpixel,
    #                                         pseudo_label_dir=pseudo_label_dir)
    # elif dataset_name == 'lc_semantic_waymo_preseg':
    #     from core.datasets.waymo_presegmentation import WaymoSparseLabelPrepare
    #     dataset = WaymoSparseLabelPrepare(root=configs.dataset.root,
    #                                       voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'waymo_eval':
    #     from core.datasets.lc_semantic_waymo_eval import LCSemanticWaymoEval
    #     dataset = LCSemanticWaymoEval(root=configs.dataset.root,
    #                                   voxel_size=configs.dataset.voxel_size,
    #                                   image_crop_rate=configs.dataset.image_crop_rate,
    #                                   sparse_label_root=configs.dataset.sparse_label_root)
    else:
        raise NotImplementedError(dataset_name)
    return dataset


def make_model(model_name=None) -> nn.Module:
    if model_name is None:
        model_name = configs.model.name
    if "cr" in configs.model:
        cr = configs.model.cr
    else:
        cr = 1.0
    if model_name == 'spvcnn':
        from core.models.semantic_kitti.spvcnn import SPVCNN
        model = SPVCNN(
            in_channel=configs.model.in_channel,
            num_classes=configs.data.num_classes,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size
        )
    # elif model_name == 'spvcnn_swiftnet34':
    #     from core.models.semantic_kitti.spvcnn_swiftnet34 import SPVCNN_SWIFTNET34
    #     model = SPVCNN_SWIFTNET34(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet34_assop2p':
    #     from core.models.semantic_kitti.spvcnn_swiftnet34_assop2p import SPVCNN_SWIFTNET34_ASSOP2P
    #     model = SPVCNN_SWIFTNET34_ASSOP2P(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         proj_channel=configs.model.proj_channel,
    #         is_estep=configs.model.is_estep
    #     )
    elif model_name == "spvcnn_swiftnet18_nusc":
        from core.models.nuscenes.spvcnn_swiftnet18 import SPVCNN_SWIFTNET18
        model = SPVCNN_SWIFTNET18(
            num_classes=configs.data.num_classes,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size,
            imagenet_pretrain=configs.model.imagenet_pretrain,
            in_channel=configs.model.in_channel,
            proj_channel=configs.model.proj_channel
        )
    elif model_name == 'spvcnn_swiftnet18_nusc_assop2p':
        from core.models.nuscenes.spvcnn_swiftnet18_nusc_assop2p import SPVCNN_SWIFTNET_ASSOP2P
        model = SPVCNN_SWIFTNET_ASSOP2P(
            num_classes=configs.data.num_classes,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size,
            imagenet_pretrain=configs.model.imagenet_pretrain
        )
    else:
        raise NotImplementedError(model_name)
    return model


def make_criterion() -> Callable:
    if configs.criterion.name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'lovasz':
        from core.criterions import MixLovaszCrossEntropy
        criterion = MixLovaszCrossEntropy(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'lc_lovasz':
        from core.criterions import MixLCLovaszCrossEntropy
        criterion = MixLCLovaszCrossEntropy(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'sparse_ce':
        from core.criterions import SparseCrossEntropy
        criterion = SparseCrossEntropy(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'sparse_lc_ce':
        from core.criterions import SparseLCCrossEntropy
        criterion = SparseLCCrossEntropy(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'sparse_pl_ce':
        from core.criterions import SparsePLCrossEntropy
        criterion = SparsePLCrossEntropy(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'sparse_assop2p_ce':
        from core.criterions import SparseAssoP2PCrossEntropy
        criterion = SparseAssoP2PCrossEntropy(ignore_index=configs.criterion.ignore_index)
    # elif configs.criterion.name == 'waymo_sparse_assop2p_ce':
    #     from core.criterions import WaymoSparseAssoP2PCrossEntropy
    #     criterion = WaymoSparseAssoP2PCrossEntropy(ignore_index=configs.criterion.ignore_index)
    # elif configs.criterion.name == 'waymo_sparse_assop2p_pl_ce':
    #     from core.criterions import WaymoSparseAssoP2PPLCrossEntropy
    #     criterion = WaymoSparseAssoP2PPLCrossEntropy(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'sparse_assop2p_pl_ce':
        from core.criterions import SparseAssoP2PPLCrossEntropy
        criterion = SparseAssoP2PPLCrossEntropy(ignore_index=configs.criterion.ignore_index)
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion

def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay,
            nesterov=configs.optimizer.nesterov)
    elif configs.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    elif configs.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.num_epochs)
    elif configs.scheduler.name == 'cosine_warmup':
        from core.schedulers import cosine_schedule_with_warmup
        from functools import partial
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                cosine_schedule_with_warmup,
                num_epochs=configs.num_epochs,
                batch_size=configs.batch_size,
                dataset_size=configs.data.training_size
            )
        )
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler
