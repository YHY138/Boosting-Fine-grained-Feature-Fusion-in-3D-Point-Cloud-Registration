import torch
from estimators.fundamental_matrix_estimator import *
from estimators.essential_matrix_estimator_stewenius import *
from loss import *
import numpy as np
import random


class GumbelSoftmaxSampler():
    """Sample based on a Gumbel-Max distribution.

    Use re-param trick for back-prop
    """
    def __init__(self, batch_size, num_samples, tau=1., device='cuda', data_type='torch.float32'):
        self.batch_size = batch_size # 这个batchsize是ransacbatchsize不是数据集的batchsize
        self.num_samples = num_samples
        # self.num_points = num_points
        self.device = device
        self.dtype = data_type
        # 下面的gumbeldist应该是论文中提到的Gumbel(0,1)分布，类似于一种正太分布
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
                torch.tensor(0., device=self.device, dtype=self.dtype),
                torch.tensor(1., device=self.device, dtype=self.dtype))
        self.tau = tau

# logits就是特征提取和匹配网络获得的匹配特征属于内点的概率值
    def sample(self, logits=None, num_points=2000, selected=None):
        # print("############### logits before", logits.shape)
        if logits==None:
            logits = torch.ones([self.batch_size, num_points], device=self.device, dtype=self.dtype, requires_grad=True)
        else:
            logits = logits.to(self.dtype).to(self.device).repeat([self.batch_size, 1]) # 训练一般调用这部分代码
            # 这部分代码是将logit，即特征匹配网络输出的匹配特征内点概率weights矩阵repeat重复复制，最终logits维度 [ransacebatchsize, point_num]
        # print("################# processed logits shape ", logits.shape, "ransacbatchsize ", self.batch_size) # 特征匹配网络进行重复复制后的shape是[ransacebatchsize, point_num]

        if selected is None:
            gumbels = self.gumbel_dist.sample(logits.shape)
            gumbels = (logits + gumbels)/self.tau # 这一步和下一步是论文中的公式5c
            y_soft = gumbels.softmax(-1)
            topk = torch.topk(gumbels, self.num_samples, dim=-1) # 的numsamples是根据论文中的solver求解需要多少个匹配特征点决定的，训练点云配准时为3
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, topk.indices, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            pass

        return ret, y_soft#, topk.indices
