import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

import finegrained_kpconv_blocks


class KPConv_MSRes(nn.Module):
    def __init__(self, in_dim, out_dim,
                 config, current_extent, radius, use_bn, bn_momentum, block_name):
        self.inplanes = in_dim
        self.out_dim = out_dim

        super(KPConv_MSRes, self).__init__()

        self.kpconv_mini = finegrained_kpconv_blocks.KPConv(7,
                             config.in_points_dim,
                             out_dim,
                             out_dim // 2,
                             current_extent,
                             radius * 1.5,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)
        self.batchnorm1 = finegrained_kpconv_blocks.BatchNormBlock(out_dim // 2, use_bn, bn_momentum)

        self.kpconv_mid = finegrained_kpconv_blocks.KPConv(13,
                          config.in_points_dim,
                          out_dim,
                          out_dim // 2,
                          current_extent,
                          radius,
                          fixed_kernel_points=config.fixed_kernel_points,
                          KP_influence=config.KP_influence,
                          aggregation_mode=config.aggregation_mode,
                          deformable='deform' in block_name,
                          modulated=config.modulated)
        self.batchnorm2 = finegrained_kpconv_blocks.BatchNormBlock(out_dim // 2, use_bn, bn_momentum)

        self.mini2mid_linear = nn.Linear(self.kpconv_mini.out_channels, self.kpconv_mini.out_channels, bias=False)
        self.mini2mid_norm = nn.BatchNorm1d(self.kpconv_mini.out_channels)

        self.midmini_linear = nn.Linear(out_dim//2, out_dim//2, bias=False)
        self.midmini_norm = nn.BatchNorm1d(out_dim//2)

        self.final_conv = nn.Linear(out_dim, out_dim, bias=False)
        self.final_norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, q_pts, s_pts, neighb_inds, x, stack_lengths_post):
        x1 = self.kpconv_mini(q_pts, s_pts, neighb_inds, x)
        x1 = self.batchnorm1(x1, stack_lengths_post)
        x2 = self.kpconv_mid(q_pts, s_pts, neighb_inds, x)
        x2 = self.batchnorm2(x2, stack_lengths_post)

        x2 = self.midmini_linear(x1 + x2)
        x = torch.cat((x1, x2), 1)

        out = self.final_conv(x)
        out = self.final_norm(out)
        self.relu(out)

        return out
