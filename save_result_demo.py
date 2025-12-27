"""Barebones code demonstrating REGTR's registration. We provide 2 demo
instances for each dataset

Simply download the pretrained weights from the project webpage, then run:
    python demo.py EXAMPLE_IDX
where EXAMPLE_IDX can be a number between 0-5 (defined at line 25)

The registration results will be shown in a 3D visualizer.
"""
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict

from typing import Union, List
import sys
sys.path.append('.')

from utils.misc import load_config
from utils.se3_numpy import se3_transform
from models import get_model
import os


def to_numpy(tensor: Union[np.ndarray, torch.Tensor, List]) -> Union[np.ndarray, List]:
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return [to_numpy(l) for l in tensor]
    elif isinstance(tensor, str):
        return tensor
    elif tensor is None:
        return None
    else:
        raise NotImplementedError
    
def load_point_cloud(fname):
    if fname.endswith('.pth'):
        data = torch.load(fname)
    elif fname.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(fname)
        data = np.asarray(pcd.points)
    elif fname.endswith('.bin'):
        data = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
    else:
        raise AssertionError('Cannot recognize point cloud format')

    return data[:, :3]  # ignore reflectance, or other features if any

def get_pca_color(feat, brightness=1.25, center=True, merge_mode='concat'):
    """
    Compute a PCA-based RGB color for point features.

    Supports input feature tensors of shape:
      - (N, C): N points, C-dim features
      - (L, N, C): L layers, N points, C-dim per-layer features

    When `feat` is 3D (L,N,C) we merge layers to a per-point feature
    according to one of the strategies below (default: 'concat'):
      - 'mean': average features across layers -> (N, C)
      - 'concat': concatenate layer channels -> (N, L*C)
      - 'select': select last layer -> (N, C)

    Returns a tensor of shape (N, 3) with values in [0,1].
    """
    # If layers dimension present (L, N, C), convert to (N, C_total)
    if feat.ndim == 3:
        # choose a default merging strategy; concat preserves maximum information
        merge_mode = getattr(get_pca_color, "merge_mode", "concat")
        if merge_mode == "mean":
            feat2 = feat.mean(dim=0)
        elif merge_mode == "select":
            # choose last layer by default
            feat2 = feat[-1]
        elif merge_mode == "concat":
            # (L, N, C) -> (N, L*C)
            L, N, C = feat.shape
            feat2 = feat.permute(1, 0, 2).reshape(N, L * C)
        else:
            raise ValueError(f"Unknown merge_mode: {merge_mode}")
    else:
        feat2 = feat

    # Ensure 2D tensor (N, D)
    if feat2.ndim != 2:
        raise ValueError("Feature tensor must be 2D (N,C) after merging layers")

    # PCA (compute up to 6 components)
    u, s, v = torch.pca_lowrank(feat2, center=center, q=6, niter=5)
    projection = feat2 @ v
    # combine first 6 components into 3 channels (mixing 1-3 and 4-6)
    # if there are fewer than 6 components keep what's available
    comps = projection.shape[1]
    if comps >= 6:
        projection_rgb = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    elif comps >= 3:
        projection_rgb = projection[:, :3]
    else:
        # pad with zeros if less than 3 components
        pad = torch.zeros((projection.shape[0], max(0, 3 - comps)), device=projection.device, dtype=projection.dtype)
        projection_rgb = torch.cat([projection, pad], dim=1)

    # normalize each channel to [0,1]
    min_val = projection_rgb.min(dim=0, keepdim=True)[0]
    max_val = projection_rgb.max(dim=0, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection_rgb - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color


def main():
    dataset_name_list = [
        '7-scenes-redkitchen',
    ]

    ckpt_path = "/home/dell/anaconda3/envs/yhy/my_RegTR/train_models/3dmatch/res2net_512regtr_scale8basewid14/ckpt/model.pth"
    # Load configuration file
    cfg = EasyDict(load_config(Path(ckpt_path).parents[1] / 'config.yaml'))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    print("#######################################################")
    print(ckpt_path)
    print("#######################################################")

    # Instantiate model and restore weights
    Model = get_model(cfg.model)
    model = Model(cfg).to(device)
    print(model._modules)
    state = torch.load(ckpt_path)
    model.load_state_dict(state['state_dict'])

    cloud_name_pre = 'cloud_bin_' # 点云文件的前缀
    cloud_name_end = '.pth'
    gtlog_path = 'gt.log'


    for dataset_name in dataset_name_list:
        print("Processing {} dataset".format(dataset_name))
        redkichen_date_path = os.path.join('/home/dell/anaconda3/envs/yhy/RegTR-main/data/data/indoor/test', dataset_name)
        file_gt = open(os.path.join(redkichen_date_path, gtlog_path), encoding="utf-8")
        gts = file_gt.readlines()

        regtr_file = open(os.path.join(redkichen_date_path, "pose.log"), "w", encoding = "utf-8")
        last_str = '{:e} {:e} {:e} {:e} \n'.format(0, 0, 0, 1)

        iteration = 1
        for i in range(0, len(gts), 5):
            # objgraph.show_most_common_types(limit=20)
            corr_ids = gts[i].split()
            print("############ corre ids ", corr_ids)
            cloud1 = cloud_name_pre + corr_ids[0] + cloud_name_end
            cloud2 = cloud_name_pre + corr_ids[1] + cloud_name_end
            src_xyz = load_point_cloud(os.path.join(redkichen_date_path,cloud1))
            tgt_xyz = load_point_cloud(os.path.join(redkichen_date_path,cloud2))

            if 'crop_radius' in cfg:
                # Crops the point cloud if necessary (set in the config file)
                crop_radius = cfg['crop_radius']
                src_xyz = src_xyz[np.linalg.norm(src_xyz, axis=1) < crop_radius, :]
                tgt_xyz = tgt_xyz[np.linalg.norm(tgt_xyz, axis=1) < crop_radius, :]

            # Feeds the data into the model
            src_xyz = torch.from_numpy(src_xyz).float().to(device)
            tgt_xyz = torch.from_numpy(tgt_xyz).float().to(device)
            data_batch = {
                'src_xyz': [src_xyz],
                'tgt_xyz': [tgt_xyz]
            }

            outputs = model(data_batch)

            b = 0
            # pose 3X4 numpy array
            pose = to_numpy(outputs['pose'][-1, b])
            # 把各算法估计的位姿保存到对应的log文件里面
            # 写ID号
            str_id = ' '.join(corr_ids)+'\n'
            regtr_file.write(str_id)

            # 记录pose
            for p in range(3):
                temp = ['{:e}'.format(x) for x in pose[p, :]]
                regtr_file.write(' '.join(temp) + '\n')

            regtr_file.write(last_str)

            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            print('**********{} cloud**********'.format(iteration))
            iteration += 1


if __name__ == '__main__':
    main()
