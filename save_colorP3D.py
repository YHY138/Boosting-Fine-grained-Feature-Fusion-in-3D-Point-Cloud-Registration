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
import sys
sys.path.append('./')

from utils.misc import load_config
from utils.se3_numpy import se3_transform
from models import get_model
import os


from typing import Union, List
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

def get_pca_color_pair(src_feat, tgt_feat, brightness=1.25, center=True, merge_mode='concat'):
    """
    Compute PCA-based RGB colors for a pair of features (source and target)
    using a shared PCA basis so that similar features across the two point
    clouds map to similar colors.
    Args:
        src_feat: Tensor of shape (N_src, C) or (L, N_src, C)
        tgt_feat: Tensor of shape (N_tgt, C) or (L, N_tgt, C)
        brightness: scalar multiplier applied after normalization
        center: whether to center data before PCA
        merge_mode: how to merge layer dimension when input is 3D: 'concat'|'mean'|'select'
    Returns:
        src_color, tgt_color: tensors of shape (N_src, 3) and (N_tgt, 3), values in [0,1]
    """
    def _merge(feat, mode):
        if feat.ndim == 3:
            L, N, C = feat.shape
            if mode == 'mean':
                return feat.mean(dim=0)
            elif mode == 'select':
                return feat[-1]
            elif mode == 'concat':
                return feat.permute(1, 0, 2).reshape(N, L * C)
            else:
                raise ValueError(f"Unknown merge mode: {mode}")
        elif feat.ndim == 2:
            return feat
        else:
            raise ValueError('Feature tensor must be 2D or 3D')

    src2 = _merge(src_feat, merge_mode)
    tgt2 = _merge(tgt_feat, merge_mode)

    # combined features to compute shared PCA basis
    combined = torch.cat([src2, tgt2], dim=0)

    # run PCA on combined set; choose q up to 6 or num dims
    q = min(6, combined.shape[1])
    u, s, v = torch.pca_lowrank(combined, center=center, q=q, niter=5)
    projection = combined @ v

    comps = projection.shape[1]
    if comps >= 6:
        projection_rgb = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    elif comps >= 3:
        projection_rgb = projection[:, :3]
    else:
        pad = torch.zeros((projection.shape[0], max(0, 3 - comps)), device=projection.device, dtype=projection.dtype)
        projection_rgb = torch.cat([projection, pad], dim=1)

    # Normalize using global min/max so src/tgt use same scale
    min_val = projection_rgb.min(dim=0, keepdim=True)[0]
    max_val = projection_rgb.max(dim=0, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color_all = (projection_rgb - min_val) / div * brightness
    color_all = color_all.clamp(0.0, 1.0)

    n_src = src2.shape[0]
    src_color = color_all[:n_src]
    tgt_color = color_all[n_src:]
    return src_color, tgt_color


def main():
    # Retrieves the model and point cloud paths
    dataset_name_list = [
        '7-scenes-redkitchen',
    ]

    ckpt_path = "/path/to/your/pretrained_model.pth" 
    
    # Load configuration file
    cfg = EasyDict(load_config(Path(ckpt_path).parents[1] / 'config.yaml'))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    print("#######################################################")
    print(ckpt_path)
    print("#######################################################")

    # Instantiate model and restore weights
    # model = RegTR(cfg).to(device)
    # 根据config文件来调用对应的Regtr文件
    Model = get_model(cfg.model)
    model = Model(cfg).to(device)
    print(model._modules)
    state = torch.load(ckpt_path)
    # print(model._modules)
    # print(state['state_dict'].keys())
    model.load_state_dict(state['state_dict'])

    cloud_name_pre = 'cloud_bin_' # 点云文件的前缀
    cloud_name_end = '.pth'
    gtlog_path = 'gt.log'

    color_path = './color_pointcloud'

    for dataset_name in dataset_name_list:
        redkichen_date_path = os.path.join('/path/to/3dmatch/indoor/test', dataset_name)
        file_gt = open(os.path.join(redkichen_date_path, gtlog_path), encoding="utf-8")
        gts = file_gt.readlines()

        method_name = "REGTR"
        color_path_dataset = os.path.join(color_path, method_name, dataset_name)

        os.makedirs(color_path_dataset, exist_ok=True)

        iteration = 1
        for i in range(0, len(gts), 5):
            # objgraph.show_most_common_types(limit=20)
            corr_ids = gts[i].split()
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

            src_feat = outputs['src_feat']
            tgt_feat = outputs['tgt_feat'] 
            src_feat_xyz = outputs['src_kp'] 
            tgt_feat_xyz = outputs['tgt_kp']
            # Use shared PCA basis across src and tgt so similar features map to similar colors
            src_color, tgt_color = get_pca_color_pair(src_feat[0], tgt_feat[0], brightness=1.25, center=True, merge_mode='concat')

            # Convert tensors to numpy arrays (float64) for Open3D and for transformation
            src_points = src_feat_xyz[0].cpu().detach().numpy().astype(np.float64)
            tgt_points = tgt_feat_xyz[0].cpu().detach().numpy().astype(np.float64)
            src_colors_np = src_color.cpu().detach().numpy().astype(np.float64)
            tgt_colors_np = tgt_color.cpu().detach().numpy().astype(np.float64)

            # Ensure color arrays have shape (N,3)
            def _ensure_rgb(arr, n_pts):
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = np.expand_dims(arr, 0)
                if arr.shape[0] != n_pts:
                    if arr.shape[0] == 1:
                        arr = np.repeat(arr, n_pts, axis=0)
                    else:
                        raise ValueError(f"Color length {arr.shape[0]} doesn't match points {n_pts}")
                if arr.shape[1] >= 3:
                    return arr[:, :3]
                else:
                    pad = np.zeros((arr.shape[0], 3 - arr.shape[1]), dtype=arr.dtype)
                    return np.concatenate([arr, pad], axis=1)

            src_colors_np = _ensure_rgb(src_colors_np, src_points.shape[0])
            tgt_colors_np = _ensure_rgb(tgt_colors_np, tgt_points.shape[0])

            # Create original pointclouds (in their native frames)
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(src_points)
            src_pcd.colors = o3d.utility.Vector3dVector(src_colors_np)
            tgt_pcd = o3d.geometry.PointCloud()
            tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points)
            tgt_pcd.colors = o3d.utility.Vector3dVector(tgt_colors_np)

            # Align source to target coordinate frame using the predicted pose
            # `pose` is a 3x4 numpy array (R|t) that maps source -> target
            try:
                src_aligned_pts = se3_transform(pose, src_points)
            except Exception:
                # if pose is torch tensor, convert
                pose_np = pose if isinstance(pose, np.ndarray) else np.asarray(pose)
                src_aligned_pts = se3_transform(pose_np, src_points)

            src_aligned_pcd = o3d.geometry.PointCloud()
            src_aligned_pcd.points = o3d.utility.Vector3dVector(src_aligned_pts.astype(np.float64))
            src_aligned_pcd.colors = o3d.utility.Vector3dVector(src_colors_np)

            pcd_last_name = corr_ids[0] + 'to' + corr_ids[1]

            # Save full original (possibly cropped) point clouds as PLY
            try:
                # `src_xyz` and `tgt_xyz` were converted to torch tensors earlier; convert back
                src_full_pts = src_xyz.cpu().detach().numpy().astype(np.float64) if isinstance(src_xyz, torch.Tensor) else np.asarray(src_xyz, dtype=np.float64)
                tgt_full_pts = tgt_xyz.cpu().detach().numpy().astype(np.float64) if isinstance(tgt_xyz, torch.Tensor) else np.asarray(tgt_xyz, dtype=np.float64)

                src_full_pcd = o3d.geometry.PointCloud()
                src_full_pcd.points = o3d.utility.Vector3dVector(src_full_pts)
                # use a neutral gray color for full clouds
                gray_src = np.tile(np.array([[0.6, 0.6, 0.6]], dtype=np.float64), (src_full_pts.shape[0], 1))
                src_full_pcd.colors = o3d.utility.Vector3dVector(gray_src)

                tgt_full_pcd = o3d.geometry.PointCloud()
                tgt_full_pcd.points = o3d.utility.Vector3dVector(tgt_full_pts)
                gray_tgt = np.tile(np.array([[0.2, 0.8, 0.2]], dtype=np.float64), (tgt_full_pts.shape[0], 1))
                tgt_full_pcd.colors = o3d.utility.Vector3dVector(gray_tgt)

                o3d.io.write_point_cloud(os.path.join(color_path_dataset, 'iter{}_src_full_{}.ply'.format(iteration, pcd_last_name)), src_full_pcd)
                o3d.io.write_point_cloud(os.path.join(color_path_dataset, 'iter{}_tgt_full_{}.ply'.format(iteration, pcd_last_name)), tgt_full_pcd)

                # Also save the full source cloud transformed into the target frame using the predicted pose
                try:
                    # pose may be numpy or torch; ensure numpy
                    pose_np = pose if isinstance(pose, np.ndarray) else np.asarray(pose)
                    src_full_aligned_pts = se3_transform(pose_np, src_full_pts)

                    src_full_aligned_pcd = o3d.geometry.PointCloud()
                    src_full_aligned_pcd.points = o3d.utility.Vector3dVector(src_full_aligned_pts.astype(np.float64))
                    # reuse same gray coloring for aligned full cloud
                    src_full_aligned_pcd.colors = o3d.utility.Vector3dVector(gray_src)

                    o3d.io.write_point_cloud(os.path.join(color_path_dataset, 'iter{}_src_full_aligned_{}.ply'.format(iteration, pcd_last_name)), src_full_aligned_pcd)
                except Exception as e:
                    print(f"Warning: failed to save aligned full source pointcloud: {e}")
            except Exception as e:
                print(f"Warning: failed to save full pointclouds: {e}")

            # Save original-frame colored pointclouds (keypoints / features)
            o3d.io.write_point_cloud(os.path.join(color_path_dataset, 'iter{}_src_{}.ply'.format(iteration, pcd_last_name)), src_pcd)
            o3d.io.write_point_cloud(os.path.join(color_path_dataset, 'iter{}_tgt_{}.ply'.format(iteration, pcd_last_name)), tgt_pcd)
            # Save aligned pointclouds (source aligned to target frame)
            o3d.io.write_point_cloud(os.path.join(color_path_dataset, 'iter{}_src_aligned_tgt_{}.ply'.format(iteration, pcd_last_name)), src_aligned_pcd)

            iteration += 1

            
if __name__ == '__main__':
    main()
