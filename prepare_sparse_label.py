import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from core import builder

from visualize_utils import SemKITTI_label_name_16


def main() -> None:
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--weight-path', default=None, type=str, help='load pretrained weight')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    dataset = builder.make_dataset()
    dataflow = dict()
    for split in dataset:
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=8,
            num_workers=16,
            shuffle=False,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    with torch.no_grad():
        lidar_token = []
        sparse_label_rate = []
        prop_label_rate = []
        neg_label_rate = []
        correct_label = np.zeros(shape=(17,))
        error_label = np.zeros(shape=(17,))
        total_label = np.zeros(shape=(17,))
        for idx, feed_dict in enumerate(tqdm(dataflow["train"])):
            lidar_token.extend(feed_dict['lidar_token'])
            for xyz, l, sp_m, pp_m, neg_m, fov in zip(feed_dict['pts'], feed_dict['targets'],
                                                      feed_dict['sparse_label_mask'], feed_dict['prop_label_mask'],
                                                      feed_dict['neg_label_mask'], feed_dict['fov_mask']):

                sparse_label_rate.append(np.sum(sp_m) / fov.shape[0])
                prop_label_rate.append(np.sum(pp_m != 0) / pp_m.shape[0])
                valid_mask = (pp_m != 0)
                l_v = l[valid_mask]
                pp_m_v = pp_m[valid_mask]
                agree_mask = (l_v == pp_m_v)
                total_label += np.bincount(l_v, minlength=17)
                for i in range(17):
                    m = (l_v == i)
                    correct_label[i] += np.sum(m & agree_mask)
                    error_label[i] += np.sum(m & (~agree_mask))
                neg_m = np.sum(neg_m, axis=-1)
                neg_label_rate.append(np.sum(neg_m != 0) / neg_m.shape[0])

            if idx != 0 and not (idx % 200):
                print('mean sparse label rate:', np.mean(sparse_label_rate))
                print('mean prop label rate:', np.mean(prop_label_rate))
                print('mean neg label rate:', np.mean(neg_label_rate))
                print(SemKITTI_label_name_16.values())
                print('correct rate:', (correct_label / total_label).tolist())
                print('error rate:', (error_label / total_label).tolist())
                print('mAcc:', np.mean((correct_label / total_label)[1:]))
                print('mErr:', np.mean((error_label / total_label)[1:]))

        print('mean sparse label rate:', np.mean(sparse_label_rate))
        print('mean prop label rate:', np.mean(prop_label_rate))
        print('mean neg label rate:', np.mean(neg_label_rate))
        print(SemKITTI_label_name_16.values())
        print('correct rate:', (correct_label / total_label).tolist())
        print('error rate:', (error_label / total_label).tolist())
        print('mAcc:', np.mean((correct_label / total_label)[1:]))
        print('mErr:', np.mean((error_label / total_label)[1:]))

if __name__ == '__main__':
    main()
