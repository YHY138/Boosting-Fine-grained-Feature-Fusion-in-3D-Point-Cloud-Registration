"""Functions for performing operations related to rigid_transformations (Torch).

Note that poses are stored in 3x4 matrices, i.e. the last row isn't stored.
Unlike otherwise stated, all functions support arbitrary batch dimensions, e.g.
poses can have shapes ([N,] 3, 4)
"""
import math
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

_EPS = 1e-6


def se3_init(rot=None, trans=None):

    assert rot is not None or trans is not None

    if rot is not None and trans is not None:
        pose = torch.cat([rot, trans], dim=-1)
    elif rot is None:  # rotation not provided: will set to identity
        pose = F.pad(trans, (3, 0))
        pose[..., 0, 0] = pose[..., 1, 1] = pose[..., 2, 2] = 1.0
    elif trans is None:  # translation not provided: will set to zero
        pose = F.pad(rot, (0, 1))

    return pose


def se3_cat(a, b):
    """Concatenates two SE3 transforms"""
    rot_a, trans_a = a[..., :3, :3], a[..., :3, 3:4]
    rot_b, trans_b = b[..., :3, :3], b[..., :3, 3:4]

    rot = rot_a @ rot_b
    trans = rot_a @ trans_b + trans_a
    dst = se3_init(rot, trans)
    return dst


def se3_inv(pose):
    """Inverts the SE3 transform"""
    rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
    irot = rot.transpose(-1, -2)
    itrans = -irot @ trans
    return se3_init(irot, itrans)


def se3_transform(pose, xyz):
    """Apply rigid transformation to points

    Args:
        pose: ([B,] 3, 4)
        xyz: ([B,] N, 3)

    Returns:

    """

    assert xyz.shape[-1] == 3 and pose.shape[:-2] == xyz.shape[:-2]

    rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
    transformed = torch.einsum('...ij,...bj->...bi', rot, xyz) + trans.transpose(-1, -2)  # Rx + t

    return transformed


def se3_transform_list(pose: Union[List[Tensor], Tensor], xyz: List[Tensor]):
    """Similar to se3_transform, but processes lists of tensors instead

    Args:
        pose: List of (3, 4)
        xyz: List of (N, 3)

    Returns:
        List of transformed xyz
    """

    B = len(xyz)
    # print("##############", B)
    # for b in range(B):
    #     print(xyz[b].shape)
    #     print(pose[b].shape)
    assert all([xyz[b].shape[-1] == 3 and pose[b].shape[:-2] == xyz[b].shape[:-2] for b in range(B)])

    transformed_all = []
    for b in range(B):
        rot, trans = pose[b][..., :3, :3], pose[b][..., :3, 3:4]
        transformed = torch.einsum('...ij,...bj->...bi', rot, xyz[b]) + trans.transpose(-1, -2)  # Rx + t
        transformed_all.append(transformed)

    return transformed_all


def inv_se3_transform_list(pose: Union[List[Tensor], Tensor], xyz: List[Tensor]):
    """Similar to se3_transform, but processes lists of tensors instead.但这里需要对pose进行求逆，xyz要和这个求逆的结果相对应

    Args:
        pose: List of (3, 4)
        xyz: List of (N, 3)

    Returns:
        List of transformed xyz
    """

    B = len(xyz)
    assert all([xyz[b].shape[-1] == 3 and pose[b].shape[:-2] == xyz[b].shape[:-2] for b in range(B)])

    transformed_all = []
    for b in range(B):
        # 位姿求逆
        rot = pose[b][..., :3, :3].T
        trans = -torch.matmul(rot, pose[b][..., :3, 3:4])
        # rot, trans = pose[b][..., :3, :3], pose[b][..., :3, 3:4]
        transformed = torch.einsum('...ij,...bj->...bi', rot, xyz[b]) + trans.transpose(-1, -2)  # Rx + t
        transformed_all.append(transformed)

    return transformed_all


def se3_compare(a, b):
    combined = se3_cat(a, se3_inv(b))

    trace = combined[..., 0, 0] + combined[..., 1, 1] + combined[..., 2, 2]
    rot_err_deg = torch.acos(torch.clamp(0.5 * (trace - 1), -1., 1.)) \
                  * 180 / math.pi
    trans_err = torch.norm(combined[..., :, 3], dim=-1)

    err = {
        'rot_deg': rot_err_deg,
        'trans': trans_err
    }
    return err

# 利用匹配点（网络预测出来的）以及这些匹配点在重叠区域内的概率值来进行位姿估计。
def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor = None):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): ([*,] N, 3) points
        b (torch.Tensor): ([*,] N, 3) points
        weights (torch.Tensor): ([*, ] N)

    Returns:
        Transform T ([*,] 3, 4) to get from a to b, i.e. T*a = b
    """

    assert a.shape == b.shape
    assert a.shape[-1] == 3

    # print("############# src points shape ", a.shape) # [x, pointnum, 3]
    # print("############# tgt points shape ", b.shape)# [x, pointnum, 3]
    # print("############# weights shape ", weights.shape)# [x, pointnum]
    # 其中x表示transformer的层数，即有6层transformer

    # zero = torch.zeros(weights[-1].size()).to(device='cuda')
    # one = torch.zeros(weights[-1].size()).to(device='cuda')
    # for i in range(weights.shape[0]):
    #     weights[i] = torch.where(weights[i] > 0.8, weights[i], zero)
    # print(weights[-1])

    if weights is not None:
        assert a.shape[:-1] == weights.shape
        assert weights.min() >= 0 and weights.max() <= 1

        weights_normalized = weights[..., None] / \
                             torch.clamp_min(torch.sum(weights, dim=-1, keepdim=True)[..., None], _EPS)
        centroid_a = torch.sum(a * weights_normalized, dim=-2)
        centroid_b = torch.sum(b * weights_normalized, dim=-2)
        a_centered = a - centroid_a[..., None, :]
        b_centered = b - centroid_b[..., None, :]
        cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)
    else:
        centroid_a = torch.mean(a, dim=-2)
        centroid_b = torch.mean(b, dim=-2)
        a_centered = a - centroid_a[..., None, :]
        b_centered = b - centroid_b[..., None, :]
        cov = a_centered.transpose(-2, -1) @ b_centered

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    # print("################## cov shape is",cov.shape) [layernum_of_tranformer, 3,3]
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[..., 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[..., None, None] > 0, rot_mat_pos, rot_mat_neg)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[..., :, None] + centroid_b[..., :, None]
    # print(translation.shape)
    transform = torch.cat((rot_mat, translation), dim=-1) # [6, 3, 4]
    # print("########### transform result shape",transform.shape)
    return transform

def sinkhorn(log_alpha, n_iters=5, slack=True):
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1
    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.
    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)
    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    zero_pad = torch.nn.ZeroPad2d((0, 1, 0, 1))
    log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

    log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

    for i in range(n_iters):
        # Row normalization
        log_alpha_padded = torch.cat((
            log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
            log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
            dim=1)

        # Column normalization
        log_alpha_padded = torch.cat((
            log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
            log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
            dim=2)

    log_alpha = log_alpha_padded[:, :-1, :-1]

    return log_alpha

def compute_rigid_transform_with_sinkhorn(xyz_s, xyz_t, affinity, slack, n_iters, mask=None):

    log_perm_matrix = sinkhorn(affinity, n_iters=n_iters, slack=slack)

    perm_matrix = torch.exp(log_perm_matrix)

    weighted_t = perm_matrix @ xyz_t / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)

    # Compute transform and transform points
    # print("##############", xyz_s.shape, weighted_t.shape) # weighted_t shape is (6, N, 3)
    # print("################", xyz_s.shape, weighted_t[-1].unsqueeze(0).shape)
    # print(torch.sum(perm_matrix, dim=2).shape)
    transform = compute_rigid_transform(xyz_s.expand(weighted_t.shape[0], -1, -1), weighted_t, weights=torch.sum(perm_matrix, dim=2)).squeeze(0)

    return transform


def fast_compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor = None, weights_threshold=0.6):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): ([*,] N, 3) points
        b (torch.Tensor): ([*,] N, 3) points
        weights (torch.Tensor): ([*, ] N)

    Returns:
        Transform T ([*,] 3, 4) to get from a to b, i.e. T*a = b
    """
    assert a.shape == b.shape
    assert a.shape[-1] == 3
    # weights_threshold = 0.6
    print("########################### 当前使用的阈值参数：", weights_threshold)

    zero = torch.zeros(weights[-1].size()).to(device='cuda')
    for i in range(weights.shape[0]):
        weights[i] = torch.where(weights[i] > weights_threshold, weights[i], zero)

    if weights is not None:
        assert a.shape[:-1] == weights.shape
        assert weights.min() >= 0 and weights.max() <= 1

        weights_normalized = weights[..., None] / \
                             torch.clamp_min(torch.sum(weights, dim=-1, keepdim=True)[..., None], _EPS)
        centroid_a = torch.sum(a * weights_normalized, dim=-2)
        centroid_b = torch.sum(b * weights_normalized, dim=-2)
        a_centered = a - centroid_a[..., None, :]
        b_centered = b - centroid_b[..., None, :]
        cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)
    else:
        centroid_a = torch.mean(a, dim=-2)
        centroid_b = torch.mean(b, dim=-2)
        a_centered = a - centroid_a[..., None, :]
        b_centered = b - centroid_b[..., None, :]
        cov = a_centered.transpose(-2, -1) @ b_centered

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    # print(cov.shape)
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[..., 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[..., None, None] > 0, rot_mat_pos, rot_mat_neg)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[..., :, None] + centroid_b[..., :, None]
    # print(translation.shape)
    transform = torch.cat((rot_mat, translation), dim=-1)
    # print(transform.shape)
    # print('))))))))))))) {}'.format(type(transform)))
    return transform