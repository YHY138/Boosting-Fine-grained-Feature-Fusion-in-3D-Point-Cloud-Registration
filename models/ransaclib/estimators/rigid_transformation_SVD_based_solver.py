import torch
import numpy as np

class RigidTransformationSVDBasedSolver:
    def __init__(self, data_type=torch.float32, device = 'cuda'):
        self.data_type = data_type
        self.device = device
        self.sample_size = 3
        self.sqrt_3 = torch.sqrt(torch.tensor(3.))

    def estimate_model(self, data, weights=None, sample_indices=None, flag=True):
        """
            https://github.com/danini/graph-cut-ransac/blob/7d4af4d4b3d5e88964631073cfb472921eb118ae/src/pygcransac/include/estimators/solver_rigid_transformation_svd.h#L92
            Now it works for a batch of data, data in a shape of [batch_size, n, 6] 这里的batchsize一般是ransacbatchsize
            output: pose in [bs, 4, 3], R, t, scale in batches
        """
        assert data.shape[-1] == 6
        # at least 3 pairs
        assert data.shape[-2] >= 3
        # if the selected indices are given
        if sample_indices is not None:
            points = torch.index_select(data, 0, sample_indices)
        else:
            points = data

        # Calculate the center of gravity for both point clouds
        # print("#################### before centroid points shape ", points.shape) # [ransacbatchsize, pointnum, 6],一般是[64,3,6]
        centroid = torch.mean(points, dim=1)
        # print("################### centroid ", centroid.shape) # [ransacbatchsize，6]
        coefficient = points - centroid[:, None, :]

        avg_distance0 = torch.sum(torch.sqrt(torch.sum(coefficient[:, :, 0:3] ** 2, dim=-1)), dim=-1) / points.shape[1]
        avg_distance1 = torch.sum(torch.sqrt(torch.sum(coefficient[:, :, 3:6] ** 2, dim=-1)), dim=-1) / points.shape[1]

        coefficients0 = (coefficient.transpose(-1, -2) * weights)[:, 0:3, :] if weights is not None else coefficient.transpose(-1, -2)[:, 0:3, :]
        coefficients1 = (coefficient.transpose(-1, -2) * weights)[:, 3:6, :] if weights is not None else coefficient.transpose(-1, -2)[:, 3:6, :]

        ratio0 = self.sqrt_3 / avg_distance0
        ratio1 = self.sqrt_3 / avg_distance1

        coefficients0 = coefficients0 * ratio0[:, None, None]
        coefficients1 = coefficients1 * ratio1[:, None, None]

        covariance = coefficients0 @ coefficients1.transpose(-1, -2)

        nan_filter = [not torch.isnan(i).any() for i in covariance]
            # covariance = covariance[nan_filter]
        # // A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
        # 				// the solution is linear subspace of dimensionality 2.
        # 				// => use the last two singular std::vectors as a basis of the space
        # 				// (according to SVD properties)
        if flag:
            u, s, v = torch.linalg.svd(covariance.transpose(-1, -2) @ covariance)
        else:
            u, s, v = torch.linalg.svd(covariance.transpose(-1, -2))
        vt = v.clone().transpose(-1, -2)
        R = vt @ u.transpose(-1, -2)

        # singularity
        mask = torch.linalg.det(R) < 0
        if mask.sum() != 0:
            vt[mask, :, 2] = -vt[mask, :, 2]
            R = vt @ u.transpose(-1, -2)

        scale = avg_distance1 / avg_distance0  # no use

        t = torch.sum(R * (-centroid[:, None, 0:3]), dim=1) + centroid[:, 3:6]

        model = torch.cat((
            torch.cat((R, t.unsqueeze(-1)), dim=-1),
            torch.tensor([[0, 0, 0, 1]], device=R.device).repeat(R.shape[0], 1, 1)
            ), dim=1
        )

        return model[nan_filter], R[nan_filter], t[nan_filter], scale[nan_filter]

# 这里的descriptor是估计出来的[R, t] 3x4维的矩阵，但是在调用函数的时候会转置一下，所以descriptor维度是 4x3
    def squared_residual(self, pts1, pts2, descriptor, threshold=0.03):
        """
            rewrite from GC-RANSAC,
            https://github.com/danini/graph-cut-ransac/blob/7d4af4d4b3d5e88964631073cfb472921eb118ae/src/pygcransac/include/estimators/rigid_transformation_estimator.h#L162

        """
        assert pts1.shape[1] == 3 # 3D points
        # homogeneous这一步把pts三维坐标变成单应坐标，即增加一维，但最后一维值是1
        pts_t = torch.cat((pts1, torch.ones((pts1.shape[0], 1), dtype=self.data_type, device= pts1.device)), dim=1) #[point_num, 4]
        # 但是因为descriptor第一个维度是ransacbatchsize，而ptst没有这个维度，所以在下面乘法完成以后，t的维度就变成了[ransacbatchsize，pointnum，3]
        t = pts_t @ descriptor # t是pts1经过R，t变换后获得的三维点坐标。 意思是，两个tensor矩阵相乘，只需要两个矩阵的倒数两个维度满足矩阵乘法关系就行了？
        # print("################### t shape ", t.shape) # [ransacbatchsize, pointnum, 3]
        squared_distance = torch.sum((pts2[None, :, :]- t) ** 2, dim=-1) # 计算经过位姿变化后的三维点与目标点之间的三维距离差距。[ransacbatchsize, pointnum]
        # print("#################### squared_distance ", squared_distance.shape)
        inlier_mask = squared_distance < threshold # 记录下模型变换点与目标点之间三维距离差异小于阈值的点的索引构建inliermask
        # print("########### inliermask value ", inlier_mask[0]) # inliermask完成阈值判断以后，会生成一个和distance维度相同的矩阵，但里面的值是True or False，即某个位置的元素是否通过阈值判断
        # 这种比大小保留索引，第一个维度是不变的，主要变得是第二个维度，即第二个维度中有多少元素满足阈值判断要求。这就相当于是，估计出ransacbatchsize个R、t，然后看每个R、t能满足多少组匹配点，从而帮助选出最好的R、t
        # print("############ squared_distance.sum(-1) ", squared_distance.sum(-1).shape) # [ransacbatchsize]
        # print("############ squared_distance.mean() ", squared_distance.mean().shape) # [] 这个应该是求整个distance矩阵的平均，不是求每个ransacbatchsize的平均
        # print("############ inlier mask ", inlier_mask.shape) # [ransacbatchsize，pointnum]
        return squared_distance.sum(-1), squared_distance.mean(), inlier_mask
