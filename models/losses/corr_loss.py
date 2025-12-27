import torch
import torch.nn as nn
import sys

sys.path.append('/home/ubuntu/anaconda3/envs/regtr/train_regtr/RegTR-main/src/utils/')
from se3_torch import se3_transform_list

_EPS = 1e-6


class CorrCriterion(nn.Module):
    """Correspondence Loss.
    """

    def __init__(self, metric='mae'):
        print("$$$$$$$$$$$$$$$$$$$$$$ CorrCriterion")
        super().__init__()
        assert metric in ['mse', 'mae']

        self.metric = metric

    def forward(self, kp_before, kp_warped_pred, pose_gt, overlap_weights=None):
        # 计算预测结果与基于真实位姿求解得到的kpgt，两个三维点在三维空间上的距离差距
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
            # print("############### Loss", overlap_weights.shape) # shape is (N)
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
    """
    计算点云1中每个点到点云2中所有点的欧氏距离矩阵

    参数:
        point_cloud1 (Tensor): 形状为 (N, 3) 的第一个点云
        point_cloud2 (Tensor): 形状为 (N, 3) 的第二个点云

    返回:
        distance_matrix (Tensor): 形状为 (N, N) 的距离矩阵，
                                distance_matrix[i][j] 表示 point_cloud1[i] 到 point_cloud2[j] 的距离
    """
    # 确保输入为浮点类型，避免整数溢出
    point_cloud1 = point_cloud1.float()
    point_cloud2 = point_cloud2.float()

    # 调整维度以便广播：(N, 1, 3) 和 (1, N, 3)
    pc1_expanded = point_cloud1.unsqueeze(1)  # 形状 [N, 1, 3]
    pc2_expanded = point_cloud2.unsqueeze(0)  # 形状 [1, N, 3]

    # 计算坐标差
    diff = pc1_expanded - pc2_expanded  # 形状 [N, N, 3]

    # 计算平方差并沿最后一个维度求和
    squared_dist = torch.sum(diff ** 2, dim=-1)  # 形状 [N, N]

    # 取平方根得到欧氏距离
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
        # 计算预测结果与基于真实位姿求解得到的kpgt，两个三维点在三维空间上的距离差距
        losses = {}
        B = pose_gt.shape[0]

        kp_warped_gt = se3_transform_list(pose_gt, kp_before)

        kp_warped_pred = torch.cat(kp_warped_pred, dim=0)
        kp_warped_gt = torch.cat(kp_warped_gt, dim=0)
        # print("############### ", len(kp_warped_gt), " ", kp_warped_gt[-1].shape) # kp_warped_gt是一个list，只有一个元素，shape为(N，3)
        corr_err = kp_warped_pred - kp_warped_gt  # shape is （N，3)是预测与真实点间的距离差向量
        # print("########### corr err", corr_err)数量级在1左右

        # 但是加了这个损失以后，显存占用会增加，暂时还不知道增加多少
        if inlier_logits is not None:
            # Loss计算方案一：
            # 下面这部分使用的是空间距离判断预测的结果是否准确。即将kp用gtPose获得kpgt，并比较kpgt和kppred在空间上的距离远近，并筛选最近邻点
            # 如果匹配很好的话，最近邻点应该就是他自己，即knn-id输出的应该是[0~N]的一个数列
            # 这里遵循的理论是：pred和gt kp的knnind应该是是[0~N]的一个数列，如果不是的话，就将对应匹配点对设置为错误匹配，即binary_mask是二值的gtcorrlabel
            # 然后inlierlogts预测是pred和gt kp是匹配点对的概率，理应和binarymask的值相近。这就是下面这个Loss的逻辑
            # dist_map = torch.sqrt(pairwise_distance(kp_warped_pred, kp_warped_gt))  # shape is (N,N)计算两组点云，两两配对点的距离
            dist_map = compute_distance_matrix(kp_warped_pred, kp_warped_gt)  # shape is (N,N)计算两组点云，两两配对点的距离
            knn_ind = torch.argmin(dist_map, dim=1)  # 找出N个点与另外哪个点距离最近 shape is （N）保留的1是每个kpwarp里的点与哪个kpwarggt里的点距离最接近
            corr_label = torch.arange(start=0, end=kp_warped_pred.shape[0], device=kp_warped_pred.device) # shape is （N）
            # 判断根据距离最近邻方法求出的corr关系索引，是否与corrlabel理论corr关系索引匹配
            binary_mask = (knn_ind == corr_label).float()
            inlier_logits = torch.cat(inlier_logits)
            inlier_err = self.inlier_cri(inlier_logits, binary_mask)
            # print('###############', inlier_err)

            # Loss计算方案二：
            # 利用空间差求平方，然后判断哪些点小于阈值，小于阈值就判断是匹配准确 sqrt是开根号操作
            # 下面使用的corr_err是否小于某个阈值计算的匹配点对label真值方法
            '''
            distance_map = torch.sum(torch.square(corr_err), dim=-1)
            # print("############", distance_map.shape) # shape is （N）
            distance_map = 1 - torch.sigmoid(distance_map / distance_map.max())
            # 根据距离大小进行0，1划分，0表示不是匹配的点
            binary_tensor = (distance_map >= 0.8).float()
            # print("################# binary_tensor", binary_tensor.shape) # shape is （N）
            inlier_logits = torch.cat(inlier_logits)
            inlier_err = self.inlier_cri(inlier_logits, binary_tensor) # 这里输入的张量的shape理应都是 （N）
            '''

            # print("$$$$$$$$ inlier_err error", inlier_err.shape) # 一个张量值 但是为什么这个张量值比REGTR原生的张量值小了3倍估计是用的损失计算方法不一样，这里是二值化损失

        if self.metric == 'mae':
            corr_err = torch.sum(torch.abs(corr_err), dim=-1)
        elif self.metric == 'mse':
            corr_err = torch.sum(torch.square(corr_err), dim=-1)
        else:
            raise NotImplementedError

        if overlap_weights is not None:
            overlap_weights = torch.cat(overlap_weights)
            # print("############### Loss", overlap_weights.shape) # shape is (N)
            mean_err = torch.sum(overlap_weights * corr_err) / torch.clamp_min(torch.sum(overlap_weights), _EPS)
        else:
            mean_err = torch.mean(corr_err, dim=1)
            # print("$$$$$$$$ mean error", mean_err.shape) # 一个张量值
            # print(mean_err)

        if inlier_logits is not None:
            # Loss计算方案三：
            # 参考下面REGTR自己采用的overlap_weights * corr_err方式来同一计算inlier损失
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

    def forward(self, logits, label):  # 这里传过来的label是标签的意思，0，1，0表示不是匹配点，1表示是。logits是你自己预测的某点对是匹配的概率
        assert torch.all(label < self.NUM_LABELS)
        loss = torch.scalar_tensor(0.).to(logits)
        for i in range(self.NUM_LABELS):
            target_mask = label == i
            if torch.any(target_mask):
                loss += self.crit(logits[target_mask], label[target_mask].to(
                    torch.float)) / self.NUM_LABELS
        return loss
