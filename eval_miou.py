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
from core.nusc_trainers import NuScenes_EM_MIOURunner
from core.callbacks import MeanIoU, PredictionSaver


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--weight-path', metavar='FILE', default=None, help='weight to be load')
    parser.add_argument('--non-dist', action='store_false', help='set to disable distributed train')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

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

    dataset = builder.make_dataset()
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
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.local_rank()])

    miou_runner = NuScenes_EM_MIOURunner(
        model=model,
        num_workers=configs.workers_per_gpu,
        seed=seed,
        weight_path=args.weight_path,
        amp_enabled=False
    )
    miou_runner.run_miou(dataflow=dataflow['val'],
                           num_epochs=1,
                           callbacks=[
                               MeanIoU(
                                   name=f'iou/val/vox',
                                   num_classes=configs.data.num_classes,
                                   ignore_label=configs.data.ignore_label,
                                   output_tensor='outputs_vox',
                                   target_tensor='targets'
                               ),
                               PredictionSaver(
                                   save_dir=os.path.dirname(args.weight_path) + '_pred'
                               )
                           ])


if __name__ == '__main__':
    main()