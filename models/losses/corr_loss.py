import torch
import torch.nn as nn
from utils.se3_torch import se3_transform_list

_EPS = 1e-6


class CorrCriterion(nn.Module):
    """Correspondence Loss.
    """

    def __init__(self, metric='mae'):
        super().__init__()
        assert metric in ['mse', 'mae']

        self.metric = metric

    def forward(self, kp_before, kp_warped_pred, pose_gt, overlap_weights=None):
        losses = {}
        B = pose_gt.shape[0]

        kp_warped_gt = se3_transform_list(pose_gt, kp_before)
        corr_err = torch.cat(kp_warped_pred, dim=0) - torch.cat(kp_warped_gt, dim=0)

        if self.metric == 'mae':
            corr_err = torch.sum(torch.abs(corr_err), dim=-1)
        elif self.metric == 'mse':
            corr_err = torch.sum(torch.square(corr_err), dim=-1)
        else:
            raise NotImplementedError

        if overlap_weights is not None:
            overlap_weights = torch.cat(overlap_weights)
            mean_err = torch.sum(overlap_weights * corr_err) / torch.clamp_min(torch.sum(overlap_weights), _EPS)
        else:
            mean_err = torch.mean(corr_err, dim=1)

        return mean_err


def pairwise_distance(
        x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances


def pdist(A, B, dist_type='L2'):
    if dist_type == 'L2':
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == 'SquareL2':
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError('Not implemented')


def compute_distance_matrix(point_cloud1, point_cloud2):
    point_cloud1 = point_cloud1.float()
    point_cloud2 = point_cloud2.float()

    # (N, 1, 3) , (1, N, 3)
    pc1_expanded = point_cloud1.unsqueeze(1)  #  [N, 1, 3]
    pc2_expanded = point_cloud2.unsqueeze(0)  #  [1, N, 3]

    diff = pc1_expanded - pc2_expanded  #  [N, N, 3]

    squared_dist = torch.sum(diff ** 2, dim=-1)  #  [N, N]

    distance_matrix = torch.sqrt(squared_dist)

    return distance_matrix


class my_CorrCriterion(nn.Module):
    """Correspondence Loss.
    """

    def __init__(self, metric='mae'):
        super().__init__()
        assert metric in ['mse', 'mae']

        self.metric = metric
        self.inlier_cri = BalancedLoss()

    def forward(self, kp_before, kp_warped_pred, pose_gt, overlap_weights=None, inlier_logits=None):
        losses = {}
        B = pose_gt.shape[0]

        kp_warped_gt = se3_transform_list(pose_gt, kp_before)

        kp_warped_pred = torch.cat(kp_warped_pred, dim=0)
        kp_warped_gt = torch.cat(kp_warped_gt, dim=0)
        corr_err = kp_warped_pred - kp_warped_gt  # shape is （N，3)

        if inlier_logits is not None:
            dist_map = compute_distance_matrix(kp_warped_pred, kp_warped_gt)  # shape is (N,N)
            knn_ind = torch.argmin(dist_map, dim=1)  # 找出N个点与另外哪个点距离最近 shape is （N）
            corr_label = torch.arange(start=0, end=kp_warped_pred.shape[0], device=kp_warped_pred.device) # shape is （N）
            binary_mask = (knn_ind == corr_label).float()
            inlier_logits = torch.cat(inlier_logits)
            inlier_err = self.inlier_cri(inlier_logits, binary_mask)

        if self.metric == 'mae':
            corr_err = torch.sum(torch.abs(corr_err), dim=-1)
        elif self.metric == 'mse':
            corr_err = torch.sum(torch.square(corr_err), dim=-1)
        else:
            raise NotImplementedError

        if overlap_weights is not None:
            overlap_weights = torch.cat(overlap_weights)
            mean_err = torch.sum(overlap_weights * corr_err) / torch.clamp_min(torch.sum(overlap_weights), _EPS)
        else:
            mean_err = torch.mean(corr_err, dim=1)

        if inlier_logits is not None:
            '''
            inlier_logits = torch.cat(inlier_logits)
            inlier_err = torch.sum(inlier_logits * corr_err) / torch.clamp_min(torch.sum(inlier_logits), _EPS)
            '''
            mean_err += inlier_err

        return mean_err


class UnbalancedLoss(nn.Module):
    NUM_LABELS = 2

    def __init__(self):
        super().__init__()
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, logits, label):
        return self.crit(logits, label.to(torch.float))


class BalancedLoss(nn.Module):
    NUM_LABELS = 2

    def __init__(self):
        super().__init__()
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, logits, label): 
        assert torch.all(label < self.NUM_LABELS)
        loss = torch.scalar_tensor(0.).to(logits)
        for i in range(self.NUM_LABELS):
            target_mask = label == i
            if torch.any(target_mask):
                loss += self.crit(logits[target_mask], label[target_mask].to(
                    torch.float)) / self.NUM_LABELS
        return loss
