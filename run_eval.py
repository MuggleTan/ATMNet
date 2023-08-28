import os
import argparse
from collections import defaultdict
import time

import torch
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import eval_parser
from model.model import GDSR, GDSRStep2
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from utils import to_cuda


class Evaluator:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.dataloader = self.get_dataloader(args)

        self.model = GDSR(args.scaling, args.dataset, args.crop_size)

        self.resume(path=args.checkpoint)
        self.model.cuda().eval()

        torch.set_grad_enabled(False)

    def evaluate(self):
        test_stats = defaultdict(float)

        for i, sample in enumerate(tqdm(self.dataloader, colour='green')):
            sample = to_cuda(sample)

            output = self.model(sample)

            _, loss_dict = self.model.get_loss(output, sample, kind='udl')

            for key in loss_dict:
                test_stats[key] += loss_dict[key]

        return {k: v / len(self.dataloader) for k, v in test_stats.items()}

    @staticmethod
    def get_dataloader(args: argparse.Namespace):
        data_args = {
            'crop_size': (args.crop_size, args.crop_size),
            'in_memory': args.in_memory,
            'max_rotation_angle': 0,
            'do_horizontal_flip': False,
            'crop_valid': True,
            'crop_deterministic': True,
            'image_transform': Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            'scaling': args.scaling
        }

        if args.dataset == 'DIML':
            depth_transform = Normalize([2749.64], [1154.29])
            dataset = DIMLDataset(os.path.join(args.data_dir, 'DIML'), **data_args, split='test',
                                  depth_transform=depth_transform)
        elif args.dataset == 'Middlebury':
            depth_transform = Normalize([2296.78], [1122.7])
            dataset = MiddleburyDataset(os.path.join(args.data_dir, 'Middlebury'), **data_args, split='test',
                                        depth_transform=depth_transform)
        elif args.dataset == 'NYUv2':
            depth_transform = Normalize([2796.32], [1386.05])
            dataset = NYUv2Dataset(os.path.join(args.data_dir, 'NYU Depth v2'), **data_args, split='test',
                                   depth_transform=depth_transform)
        else:
            raise NotImplementedError(f'Dataset {args.dataset}')

        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                          drop_last=False)

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        checkpoint = torch.load(path, 'cuda:{}'.format(args.cuda))
        if 'model' in checkpoint:
            self.model = GDSRStep2(args.scaling, args.dataset, args.crop_size, cuda=args.cuda)
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = eval_parser.parse_args()
    print(eval_parser.format_values())

    torch.cuda.set_device(args.cuda)
    evaluator = Evaluator(args)

    since = time.time()
    stats = evaluator.evaluate()
    time_elapsed = time.time() - since

    # de-standardize losses and convert to cm (cm^2, respectively)
    std = evaluator.dataloader.dataset.depth_transform.std[0]
    stats['l1_loss'] = 0.1 * std * stats['l1_loss']
    stats['mse_loss'] = 0.01 * std ** 2 * stats['mse_loss']

    print('Evaluation completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(stats)
