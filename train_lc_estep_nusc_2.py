import argparse
import os
import sys

import random
import numpy as np

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir, get_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.nusc_trainers import NuScenesLCEstepRunner2
from core.callbacks import PseudoLabelEvaluator, PseudoLabelSaver, PseudoLabelVisualizer, ConfuseMatrix, SemKITTI_label_name_16
from core.models.utils import SparseSyncBatchNorm


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--weight-path', metavar='FILE', default=None, help='weight to be load')
    parser.add_argument('--non-dist', action='store_false', help='set to disable distributed train')
    parser.add_argument('--pseudo-label-dir', metavar='DIR', default=None, help='pseudo label dir')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    configs.update(vars(args))

    if args.non_dist:
        dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2 ** 32 - 1)

    seed = configs.train.seed + dist.rank() * configs.workers_per_gpu * configs.num_epochs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    run_dir = get_run_dir()
    c_it = 0
    for d in os.listdir(run_dir):
        if 'checkpoints_mstep' in d:
            c_it += 1
    pseudo_label_dir = args.pseudo_label_dir
    if pseudo_label_dir is None:
        pseudo_label_dir = os.path.join(run_dir, 'pseudo_labels_' + str(c_it-1))
    print("pseudo_label_dir:", pseudo_label_dir)

    dataset = builder.make_dataset(pseudo_label_dir=pseudo_label_dir)
    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=False)
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    model = builder.make_model().cuda()
    if args.non_dist:
        model = SparseSyncBatchNorm.convert_sync_batchnorm(model)
    print(model)
    if args.non_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[dist.local_rank()])

    estep_runner = NuScenesLCEstepRunner2(
        model=model,
        num_workers=configs.workers_per_gpu,
        seed=seed,
        weight_path=args.weight_path,
        amp_enabled=False
    )

    estep_runner.run_estep(dataflow=dataflow['train'],
                           num_epochs=1,
                           callbacks=[
                               # PseudoLabelVisualizer(dataset='nusc'),
                               PseudoLabelEvaluator(
                                   num_classes=configs.data.num_classes,
                                   ignore_label=configs.data.ignore_label
                               ),
                               PseudoLabelSaver(),
                               ConfuseMatrix(num_classes=configs.data.num_classes,
                                             ignore_label=configs.data.ignore_label,
                                             xbar_names=list(SemKITTI_label_name_16.values())[1:],
                                             fig_size=(16, 16))
                           ])


if __name__ == '__main__':
    main()
