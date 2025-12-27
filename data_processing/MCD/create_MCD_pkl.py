'''
This code converts the MCD dataset into a PKL file format consistent with 3DMatch, 
based on the PCD files corresponding to each pose,
and requires computing the relative pose relationship between two PCD files.
'''
import os, sys, glob, time
import numpy as np
from scipy.spatial.transform import Rotation

import open3d as o3d
import pickle

import torch

from compute_MCDoverlap import cal_overlap_twoPCD

# Folder to export the pointclouds to
exported_dir = '/path/to/your/MCD/dataset'
pth_files = sorted(glob.glob(exported_dir + '/cloud_inBody_mergeFrameSample/*.pth'))
pose_files = sorted(glob.glob(exported_dir + '/cloud_inBody_mergeFrameSample/*.txt'))
src, tgt, rot, trans, overlap = [], [], [], [], []
pcd_number = len(pth_files)
total_num = 0 
train_ratio = 0.8
for i in range(pcd_number):

    for j in range(i+1, pcd_number):
        if i == j:
            continue
        src_files, tgt_files = pth_files[i], pth_files[j]
        pose_path1, pose_path2 = pose_files[i], pose_files[j]
        pcd_numpy1 = torch.load(src_files)
        pcd_numpy2 = torch.load(tgt_files)
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

        np_pcd1 = np.asarray(pcd1.points)
        np_pcd2 = np.asarray(pcd2.points)
        np_pcd1 = np.dot(T1[:3,:3], np_pcd1.T).T + T1[:3,3]
        np_pcd2 = np.dot(T2[:3,:3], np_pcd2.T).T + T2[:3,3]
        pc_o3d1 = o3d.geometry.PointCloud()
        pc_o3d1.points = o3d.utility.Vector3dVector(np_pcd1)
        pc_o3d1.paint_uniform_color([0, 1, 0])
        pc_o3d2 = o3d.geometry.PointCloud()
        pc_o3d2.points = o3d.utility.Vector3dVector(np_pcd2)
        pc_o3d2.paint_uniform_color([0, 0, 1])

        R_2_1 = np.dot(R_2_W, T1[:3,:3])
        t_2_1 = np.dot(R_2_W, T1[:3,3])+t_2_W

        overlap_ratio = cal_overlap_twoPCD(pc_o3d1, pc_o3d2, 0.5)
        overlap.append(overlap_ratio)
        src.append(src_files)
        tgt.append(tgt_files)
        rot.append(R_2_1)
        trans.append(t_2_1.reshape(3,1))
        total_num += 1

mcd_pkl = {}
mcd_pkl['src'] = np.array(src[:int(total_num*train_ratio)])
mcd_pkl['tgt'] = np.array(tgt[:int(total_num*train_ratio)])
mcd_pkl['overlap'] = np.array(overlap[:int(total_num*train_ratio)])
mcd_pkl['rot'] = np.array(rot[:int(total_num*train_ratio)])
mcd_pkl['trans'] = np.array(trans[:int(total_num*train_ratio)])
val_mcdpkl = {}
val_mcdpkl['src'] = np.array(src[int(total_num*train_ratio):])
val_mcdpkl['tgt'] = np.array(tgt[int(total_num*train_ratio):])
val_mcdpkl['overlap'] = np.array(overlap[int(total_num*train_ratio):])
val_mcdpkl['rot'] = np.array(rot[int(total_num*train_ratio):])
val_mcdpkl['trans'] = np.array(trans[int(total_num*train_ratio):])

print(mcd_pkl.keys())
print(mcd_pkl['overlap'].shape)#<class 'numpy.ndarray'>(n,)
print(type(mcd_pkl['overlap']))
print(mcd_pkl['rot'].shape)# <class 'numpy.ndarray'> (n, 3,3)
print(type(mcd_pkl['rot']))
print(mcd_pkl['trans'].shape)# <class 'numpy.ndarray'> （n, 3, 1)

file = open('./tuhh_night_09_os1_64-001_mergeMini_train.pkl', 'wb')
pickle.dump(mcd_pkl, file)
file.close()

val_file = open('./tuhh_night_09_os1_64-001_mergeMini_val.pkl', 'wb')
pickle.dump(val_mcdpkl, val_file)
val_file.close()


