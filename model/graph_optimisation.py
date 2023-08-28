from abc import ABC

import numpy as np
import torch
import cupy as cp

from cupyx.scipy.sparse.linalg import cg

import scipy.sparse as sp
import torch.nn.functional as F


def get_neighbor_affinity_no_border(feature_map, mu, lambda_):
    B, M, H, W = feature_map.shape  # batch_size, channels, height, width

    # 先在四个方向加一层padding，方便后续处理
    feature_map_padded = F.pad(feature_map, (1, 1, 1, 1), 'constant', 0)

    # 计算像素与邻域像素平方差，并计算通道维度均值
    top = torch.mean((feature_map_padded[:, :, 0:-2, 1:-1] - feature_map) ** 2, dim=1, keepdim=True)
    bottom = torch.mean((feature_map_padded[:, :, 2:, 1:-1] - feature_map) ** 2, dim=1, keepdim=True)
    left = torch.mean((feature_map_padded[:, :, 1:-1, 0:-2] - feature_map) ** 2, dim=1, keepdim=True)
    right = torch.mean((feature_map_padded[:, :, 1:-1, 2:] - feature_map) ** 2, dim=1, keepdim=True)

    affinity = torch.cat([top, bottom, left, right], dim=1) / (1e-6 + mu ** 2)
    affinity = torch.exp(-affinity)

    # 构造一个矩阵，去除边缘像素
    border_remover = torch.ones((1, 4, H, W), device=feature_map.device)
    border_remover[0, 0, 0, :] = 0  # top
    border_remover[0, 1, -1, :] = 0  # bottom
    border_remover[0, 2, :, 0] = 0  # left
    border_remover[0, 3, :, -1] = 0  # right

    affinity = affinity * border_remover
    # 与邻域像素的差之和
    center = torch.sum(affinity, dim=1, keepdim=True)
    affinity = torch.cat([affinity, center], dim=1)
    # 加权
    affinity = affinity * lambda_

    return affinity


MAX_ITER = 1500


def create_fixed_cupy_sparse_matrices(H, W, upsampling):
    # H == W == crop_size * crop_size == 256 * 256       upsampling == scaling
    h = H // upsampling
    w = W // upsampling

    # create the mapping matrix from neighbor dense affinity to sparse Laplacian
    matrices = {}
    for location in ('top', 'bottom', 'left', 'right'):
        indices = np.zeros((4, H * W - W), dtype=int)
        l = 0
        for i, j in np.ndindex(H, W):
            if location == 'top' and i > 0:
                indices[:, l] = np.array([i, j, i - 1, j], dtype=int)
            elif location == 'bottom' and i < H - 1:
                indices[:, l] = np.array([i, j, i + 1, j], dtype=int)
            elif location == 'left' and j > 0:
                indices[:, l] = np.array([i, j, i, j - 1], dtype=int)
            elif location == 'right' and j < W - 1:
                indices[:, l] = np.array([i, j, i, j + 1], dtype=int)
            else:
                continue

            l += 1

        assert l == H * W - W
        indices_a, indices_b = np.unravel_index(np.ravel_multi_index(indices, (H, W, H, W)), (H * W, H * W))
        matrices[f'remap_{location}'] = cp.sparse.coo_matrix(sp.coo_matrix(
            (-1 * np.ones(len(indices_a), dtype=np.float32), (indices_a, indices_b)), shape=(H * W, H * W))).tocsr()

    matrices['remap_center'] = cp.sparse.eye(H * W, format='csr', dtype=np.float32)
    M = -matrices['remap_top'] - matrices['remap_bottom'] - matrices['remap_left'] - matrices['remap_right'] \
        + matrices['remap_center']

    # create array for downsampling
    D = cp.zeros((h * w, h, w), dtype=np.float32)
    for i, j in np.ndindex(h, w):
        D[i * w + j, i, j] = 1
    D = cp.kron(D, cp.ones((1, upsampling, upsampling), dtype=np.float32)) / (upsampling ** 2)  # h*w x H x W
    D = cp.sparse.coo_matrix(D.reshape((h * w, H * W))).tocsr()
    DtD = D.transpose().dot(D)

    return {**matrices, 'M': M, 'D': D, 'DtD': DtD}


class GraphOptimization(torch.autograd.Function):
    """
    Solves the problem min_x ||Dx - x_lr||^2 + lambda x^T L x, propagating gradients through the laplacian L.
    """
    @staticmethod
    def forward(ctx, neighbor_affinity, source, fixed_matrices, mask_source=None):
        """
        neighbor_affinity (B x 5 x H x W): affinity among neighbor pixels
        source (B x 1 x h x w): source image
        mask_source (B x 1 x h x w): mask source image
        """
        assert neighbor_affinity.is_cuda
        assert source.is_cuda
        assert source.shape[1] == 1
        assert not source.requires_grad
        # batch: 8 channel: 1  h = w = crop_size / scale
        B, _, h, w = source.shape
        # 4096 * 65536
        D = fixed_matrices['D']
        # 8 * 5 * 256 * 256
        neighbor_affinity_cp = cp.asarray(neighbor_affinity.detach())
        # 8 * 4096 * 1
        source_cp = cp.asarray(source.detach().reshape((B, -1, 1)))

        if mask_source is not None:
            mask_source_cp = cp.asarray(mask_source.detach().reshape((B, -1)))
        else:
            # 8 * 4096
            mask_source_cp = cp.ones((B, h * w))

        As = []
        bs = []
        for idx in range(0, B):
            # 65536 * 65536
            L = build_laplacian(neighbor_affinity_cp[idx:idx + 1], fixed_matrices)
            # 4096 * 4096
            C = cp.sparse.diags(mask_source_cp[idx]).tocsr()
            # 4096 * 65536
            CD = C.dot(D)
            # 65536 * 65536
            DtCD = D.transpose().dot(CD)

            As.append((DtCD + L))
            bs.append(CD.transpose().dot(source_cp[idx]))

        A = [[None] * B for _ in range(B)]
        for i in range(B):
            A[i][i] = As[i]
        A = cp.sparse.bmat(A).tocsr()
        b = cp.concatenate(bs, axis=0)

        x_cp = cg(A, b, maxiter=MAX_ITER)[0]
        x_cp = x_cp.reshape((B, -1, 1))
        x = torch.as_tensor(x_cp, device='cuda')
        x = x.reshape((B, 1, neighbor_affinity.shape[2], neighbor_affinity.shape[3]))

        if ctx.needs_input_grad[0]:
            x.requires_grad = True

        ctx.x_cp = x_cp
        ctx.fixed_matrices = fixed_matrices
        ctx.A = A
        ctx.neighbor_affinity_shape = neighbor_affinity.shape

        return x

    @staticmethod
    def backward(ctx, grad_x):
        if grad_x is None:
            return None, None, None, None

        assert grad_x.shape[1] == 1

        x_cps = ctx.x_cp
        fixed_matrices = ctx.fixed_matrices
        A = ctx.A
        B, _, H, W = grad_x.shape

        grad_x_cp = cp.asarray(grad_x.detach())
        grad_x_cp = grad_x_cp.reshape((-1, 1))
        grad_b_cp = cg(A.transpose().tocsr(), grad_x_cp, maxiter=MAX_ITER)[0]
        grad_b_cp = grad_b_cp.reshape((B, -1, 1))

        grad_neighbor_affinities = []
        for idx in range(0, B):
            grad_neighbor_affinity_cp = cp.zeros((5, A.shape[0] // B))

            for i, k in enumerate(('remap_top', 'remap_bottom', 'remap_left', 'remap_right')):
                grad_neighbor_affinity_cp[i] = (cp.sparse.diags(-grad_b_cp[idx].squeeze()).dot(
                    fixed_matrices[k].transpose().dot(cp.sparse.diags(x_cps[idx].squeeze())))).sum(axis=0).squeeze()

            grad_neighbor_affinity_cp[4] = (cp.sparse.diags(-grad_b_cp[idx].squeeze()).dot(
                fixed_matrices['remap_center'].dot(cp.sparse.diags(x_cps[idx].squeeze())))).sum(axis=0).squeeze()
            grad_neighbor_affinity = torch.as_tensor(grad_neighbor_affinity_cp, device='cuda')
            grad_neighbor_affinities.append(grad_neighbor_affinity.reshape((1, 5, H, W)))

        grad_neighbor_affinity = torch.cat(grad_neighbor_affinities, dim=0)
        return grad_neighbor_affinity, None, None, None


def build_laplacian(neighbor_affinity, remap):
    # 1 * 5 * 256 * 256
    _, _, H, W = neighbor_affinity.shape
    neighbor_affinity = neighbor_affinity.reshape(5, H * W)

    return cp.sparse.diags(neighbor_affinity[0]).dot(remap['remap_top']) + \
           cp.sparse.diags(neighbor_affinity[1]).dot(remap['remap_bottom']) + \
           cp.sparse.diags(neighbor_affinity[2]).dot(remap['remap_left']) + \
           cp.sparse.diags(neighbor_affinity[3]).dot(remap['remap_right']) + \
           cp.sparse.diags(neighbor_affinity[4]).dot(remap['remap_center'])
