from math import log

import torch
from torch import nn

from losses import l1_loss_func, mse_loss_func
from .extractor import MultiModalFeatureExtractor, VarEstimateModule
from .graph_optimisation import create_fixed_cupy_sparse_matrices, get_neighbor_affinity_no_border, GraphOptimization


class GDSR(nn.Module):
    def __init__(self, scaling_factor: int, dataset: str, crop_size=256, lambda_init=1.0, mu_init=0.1):
        super(GDSR, self).__init__()

        if crop_size not in [64, 128, 256]:
            raise ValueError('Crop size should be in {64, 128, 256}, got ' + str(crop_size))

        self.feature_extractor = MultiModalFeatureExtractor()

        self.log_lambda = nn.Parameter(torch.tensor([log(lambda_init)]))
        self.log_mu = nn.Parameter(torch.tensor([log(mu_init)]))

        self.mx_dict = create_fixed_cupy_sparse_matrices(crop_size, crop_size, scaling_factor)

        self.var_conv = VarEstimateModule(in_channel=4, out_channel=1, kernel_size=3)

    def forward(self, sample, only_var=False):
        guide, source, mask_lr = sample['guide'], sample['source'], sample['mask_lr']

        var = self.var_conv(torch.cat([guide, sample['y_bicubic']], dim=1))

        if only_var:
            return var

        mu, lambda_ = torch.exp(self.log_mu), torch.exp(self.log_lambda)

        features = self.feature_extractor(guide, sample['y_bicubic'])

        # batch * 5 * 256 * 256
        neighbor_affinity = get_neighbor_affinity_no_border(features, mu, lambda_)

        # 8 * 1 * 256 * 256  327680
        y_pred = GraphOptimization.apply(neighbor_affinity, source, self.mx_dict, mask_lr)

        return {'y_pred': y_pred, 'var': var, 'neighbor_affinity': neighbor_affinity}

    def get_loss(self, output, sample, kind):
        # 8 * 1 * 256 * 256
        y_pred = output['y_pred']
        # 8 * 1 * 256 * 256
        var = output['var']
        # y: ground_truth  8 * 1 * 256 * 256   mask_hr: 8 * 1 * 256 * 256
        y, mask_hr = sample['y'], sample['mask_hr']
        # ESU  estimate sparse uncertainty
        s = torch.exp(-var)
        sr_ = torch.mul(y_pred, s)
        hr_ = torch.mul(y, s)
        udl_loss = l1_loss_func(sr_, hr_, mask_hr) + 2 * torch.mean(torch.abs(var))

        l1_loss = l1_loss_func(y_pred, y, mask_hr)
        mse_loss = mse_loss_func(y_pred, y, mask_hr)

        if kind == 'l1':
            loss = l1_loss
        elif kind == 'mse':
            loss = mse_loss
        elif kind == 'udl':
            loss = udl_loss
        else:
            raise RuntimeError('No such loss!')

        return loss, {
            'l1_loss': l1_loss.detach().item(),
            'udl_loss': udl_loss.detach().item(),
            'mse_loss': mse_loss.detach().item(),
            'mu': torch.exp(self.log_mu).detach().item(),
            'lambda': torch.exp(self.log_lambda).detach().item(),
            'optimization_loss': loss.detach().item(),
            'average_link': torch.mean(output['neighbor_affinity'][:, 0:4].detach()).item()
        }


class GDSRStep2(nn.Module):
    def __init__(self, scaling_factor: int, dataset: str, crop_size=256, load_path='.', cuda=0):
        super().__init__()
        self.GDSR_var = GDSR(scaling_factor, dataset, crop_size)
        self.GDSR_U = GDSR(scaling_factor, dataset, crop_size)
        if load_path != '.':
            checkpoint = torch.load(load_path, 'cuda:{}'.format(cuda))
            self.GDSR_var.load_state_dict(checkpoint['model'])
            self.GDSR_U.load_state_dict(checkpoint['model'])

    def forward(self, sample):
        with torch.no_grad():
            var = self.GDSR_var(sample, only_var=True)
        output = self.GDSR_U(sample)
        output['var'] = var

        return output

    def get_loss(self, output, sample, kind):
        # 8 * 1 * 256 * 256
        y_pred = output['y_pred']
        # 8 * 1 * 256 * 256
        var = output['var']

        # y: ground_truth  8 * 1 * 256 * 256   mask_hr: 8 * 1 * 256 * 256  mask_lr: 8 * 1 * 32 * 32
        y, mask_hr = sample['y'], sample['mask_hr']
        # UDL uncertainty loss
        b, c, h, w = var.shape
        s1 = var.view(b, c, -1)

        pmin = torch.min(s1, dim=-1)
        pmin = pmin[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        s = var
        s = s - pmin + 1
        sr_ = torch.mul(y_pred, s)
        hr_ = torch.mul(y, s)
        udl_loss = l1_loss_func(sr_, hr_, mask_hr)

        l1_loss = l1_loss_func(y_pred, y, mask_hr)
        mse_loss = mse_loss_func(y_pred, y, mask_hr)

        if kind == 'l1':
            loss = l1_loss
        elif kind == 'mse':
            loss = mse_loss
        elif kind == 'udl':
            loss = udl_loss
        else:
            raise RuntimeError('No such loss!')

        return loss, {
            'l1_loss': l1_loss.detach().item(),
            'udl_loss': udl_loss.detach().item(),
            'mse_loss': mse_loss.detach().item(),
            'mu': torch.exp(self.GDSR_U.log_mu).detach().item(),
            'lambda': torch.exp(self.GDSR_U.log_lambda).detach().item(),
            'optimization_loss': loss.detach().item(),
            'average_link': torch.mean(output['neighbor_affinity'][:, 0:4].detach()).item()
        }
