import os, sys, glob, time
import numpy as np

# python wrapper for spline
from ceva import Ceva
# pcd interface
from pypcd import pypcd
import open3d as o3d
from tqdm import tqdm
import pickle

import torch


def load_point_cloud(fname):
    if fname.endswith('.pth'):
        data = torch.load(fname)
    elif fname.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(fname)
        data = np.asarray(pcd.points)
    elif fname.endswith('.bin'):
        data = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
    elif fname.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(fname)
        data = np.asarray(pcd.points)
    else:
        raise AssertionError('Cannot recognize point cloud format')

    return data[:, :3]  # ignore reflectance, or other features if any

from compute_MCDoverlap import cal_overlap_twoPCD

# Folder to export the pointclouds to
exported_dir = 'your_exported_dir'
pth_files = sorted(glob.glob(exported_dir + '/cloud_inBody_mergeFrame_wGap/*.pcd'))
pose_files = sorted(glob.glob(exported_dir + '/cloud_inBody_mergeFrame_wGap/*.txt'))
gtlog_files = open(exported_dir + '/cloud_inBody_mergeFrame_wGapSample/'+'gt.log', 'w')
last_str = '{:e} {:e} {:e} {:e} \n'.format(0, 0, 0, 1)

src, tgt, rot, trans, overlap = [], [], [], [], []
pcd_number = len(pth_files)
total_num = 0 
train_ratio = 1
for i in range(pcd_number):
    for j in range(i+4, i+6):
        if not j < pcd_number:
            continue
        src_files, tgt_files = pth_files[i], pth_files[j]
        pose_path1, pose_path2 = pose_files[i], pose_files[j]
        pcd_numpy1 = load_point_cloud(src_files)
        pcd_numpy2 = load_point_cloud(tgt_files)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcd_numpy1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcd_numpy2)
        pose_file1 = open(pose_path1, 'r',encoding='utf-8')
        pose1 = pose_file1.readlines()
        pose_file2 = open(pose_path2, 'r',encoding='utf-8')
        pose2 = pose_file2.readlines()
        T1, T2 = np.zeros([4,4]), np.zeros([4,4])
        for k in range(3):
            T1[k] = [float(r) for r in pose1[k].split()]
            T2[k] = [float(r) for r in pose2[k].split()]
        T1[3,3] = 1
        T2[3,3] = 1

        R_2_W = T2[:3, :3].T
        t_2_W = -1 * T2[:3, :3].T @ T2[:3,3]
        np_pcd1_insrc = np.asarray(pcd1.points)
        np_pcd2_intgt = np.asarray(pcd2.points)
        np_pcd1_inW = np.dot(T1[:3,:3], np_pcd1_insrc.T).T + T1[:3,3]
        np_pcd2_inW = np.dot(T2[:3,:3], np_pcd2_intgt.T).T + T2[:3,3]
        pc_o3d1 = o3d.geometry.PointCloud()
        pc_o3d1.points = o3d.utility.Vector3dVector(np_pcd1_inW)
        pc_o3d2 = o3d.geometry.PointCloud()
        pc_o3d2.points = o3d.utility.Vector3dVector(np_pcd2_inW)

        R_2_1 = np.dot(R_2_W, T1[:3,:3])
        t_2_1 = np.dot(R_2_W, T1[:3,3])+t_2_W

        np_pcd1_intgt = np.dot(R_2_1, pcd_numpy1.T).T + t_2_1
        pcd1_in_pcd2 = o3d.geometry.PointCloud()
        pcd1_in_pcd2.points = o3d.utility.Vector3dVector(np_pcd1_intgt)
        pcd1_in_pcd2.paint_uniform_color([0, 0, 1])
        pcd1.paint_uniform_color([1, 0, 0]) 
        pcd2.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([pcd2, pcd1_in_pcd2, pcd1])
     
gtlog_files.close()

mcd_pkl = {}
mcd_pkl['src'] = np.array(src[:int(total_num*train_ratio)])
mcd_pkl['tgt'] = np.array(tgt[:int(total_num*train_ratio)])
mcd_pkl['overlap'] = np.array(overlap[:int(total_num*train_ratio)])
mcd_pkl['rot'] = np.array(rot[:int(total_num*train_ratio)])
mcd_pkl['trans'] = np.array(trans[:int(total_num*train_ratio)])

file = open(exported_dir + '/temp.pkl', 'wb')
pickle.dump(mcd_pkl, file)
file.close()



