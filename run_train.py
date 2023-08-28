import os
import argparse
from collections import defaultdict
import time

import numpy as np
import torch

from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import tqdm

from arguments import train_parser
from model.model import GDSR, GDSRStep2
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from utils import new_log, to_cuda, seed_all

class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.dataloaders = self.get_dataloaders(args)

        seed_all(args.seed)

        if args.step2:
            self.model = GDSRStep2(args.scaling, args.dataset, args.crop_size, load_path=args.step1_path, cuda=args.cuda)
        else:
            self.model = GDSR(args.scaling, args.dataset, args.crop_size)

        self.scaler = GradScaler()
        self.model.cuda()

        if args.resume is None:
            self.experiment_folder = new_log(os.path.join(args.save_dir, 'x' + str(args.scaling), args.dataset), args)
        else:
            self.experiment_folder = os.path.join(args.save_dir, 'x' + str(args.scaling), args.dataset, args.resume.split('/')[-2])

        self.writer = SummaryWriter(log_dir=self.experiment_folder)

        dataset = args.dataset

        if dataset == 'DIML':
            self.lr_step = 6
            if args.step2:
                self.epochs = 100
            else:
                self.epochs = 100
        elif dataset == 'Middlebury':
            self.lr_step = 100
            if args.step2:
                self.epochs = 1500
            else:
                self.epochs = 1500
        elif dataset == 'NYUv2':
            self.lr_step = 10
            if args.step2:
                self.epochs = 150
            else:
                self.epochs = 150

        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=self.args.momentum,
                                       weight_decay=args.w_decay)

        if args.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step, gamma=args.lr_gamma)
        elif args.lr_scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.lr_step,
                                                                  factor=args.lr_gamma)
        else:
            self.scheduler = None

        self.epoch = 0
        self.iter = 0
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

        if args.resume is not None:
            self.resume(path=args.resume)

    def train(self):
        with tqdm(range(self.epoch, self.epochs), colour='green', leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')

                if self.args.lr_scheduler == 'step':
                    self.scheduler.step()

                    self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()), self.epoch)

                self.epoch += 1

    def train_epoch(self, tnr=None):
        self.train_stats = defaultdict(float)

        self.model.train()

        with tqdm(self.dataloaders['train'], colour='green', leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)
            for i, sample in enumerate(inner_tnr):

                sample = to_cuda(sample)
                self.optimizer.zero_grad()

                with autocast():
                    output = self.model(sample)
                    loss, loss_dict = self.model.get_loss(output, sample, kind=self.args.loss)

                for key in loss_dict:
                    self.train_stats[key] += loss_dict[key]

                if self.epoch > 0 or not self.args.skip_first:

                    self.scaler.scale(loss).backward()

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.iter += 1

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    self.train_stats = {k: v / self.args.logstep_train for k, v in self.train_stats.items()}

                    inner_tnr.set_postfix(training_loss=self.train_stats['optimization_loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['optimization_loss'],
                                        validation_loss=self.val_stats['optimization_loss'],
                                        best_validation_loss=self.best_optimization_loss)

                    for key in self.train_stats:
                        self.writer.add_scalar('train/' + key, self.train_stats[key], self.iter)

                    # reset metrics
                    self.train_stats = defaultdict(float)

    def validate(self):
        self.val_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.dataloaders['val'], colour='green', leave=False):
                sample = to_cuda(sample)

                output = self.model(sample)
                loss, loss_dict = self.model.get_loss(output, sample, kind=self.args.loss)

                for key in loss_dict:
                    self.val_stats[key] += loss_dict[key]

            self.val_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}

            for key in self.val_stats:
                self.writer.add_scalar('val/' + key, self.val_stats[key], self.epoch)

            if self.val_stats['optimization_loss'] < self.best_optimization_loss:
                self.best_optimization_loss = self.val_stats['optimization_loss']
                if self.args.save_model in ['best', 'both']:
                    self.save_model('best')

    @staticmethod
    def get_dataloaders(args):
        data_args = {
            'crop_size': (args.crop_size, args.crop_size),
            'in_memory': args.in_memory,
            'max_rotation_angle': args.max_rotation,
            'do_horizontal_flip': not args.no_flip,
            'crop_valid': True,
            'image_transform': Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            'scaling': args.scaling
        }

        phases = ('train', 'val')
        if args.dataset == 'Middlebury':
            depth_transform = Normalize([2296.78], [1122.7])
            datasets = {phase: MiddleburyDataset(os.path.join(args.data_dir, 'Middlebury'), **data_args, split=phase,
                                                 depth_transform=depth_transform, crop_deterministic=phase == 'val') for
                        phase in phases}

        elif args.dataset == 'DIML':
            depth_transform = Normalize([2749.64], [1154.29])
            datasets = {phase: DIMLDataset(os.path.join(args.data_dir, 'DIML'), **data_args, split=phase,
                                           depth_transform=depth_transform) for phase in phases}

        elif args.dataset == 'NYUv2':
            depth_transform = Normalize([2796.32], [1386.05])
            datasets = {phase: NYUv2Dataset(os.path.join(args.data_dir, 'NYU Depth v2'), **data_args, split=phase,
                                            depth_transform=depth_transform) for phase in phases}
        else:
            raise NotImplementedError(f'Dataset {args.dataset}')

        return {phase: DataLoader(datasets[phase], batch_size=args.batch_size, num_workers=args.num_workers,
                                  shuffle=True) for phase in phases}

    def save_model(self, prefix=''):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch + 1,
            'iter': self.iter
        }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        checkpoint = torch.load(path, 'cuda:{}'.format(args.cuda))

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.iter = checkpoint['iter']

        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = train_parser.parse_args()
    print(train_parser.format_values())

    torch.cuda.set_device('cuda:{}'.format(args.cuda))
    trainer = Trainer(args)

    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
