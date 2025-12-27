'''
This code is designed to convert the MCD dataset into a PKL file format similar to that of 3DMatch, 
based on the PCD files corresponding to each pose, and it requires computing the relative pose relationship between two PCD files.
'''
import os, sys, glob, time
import numpy as np

import open3d as o3d
import shutil

exported_dir = 'your_exported_dir'
pcd_files = sorted(glob.glob(exported_dir + '/cloud_inBody/*.pcd'))
pose_files = sorted(glob.glob(exported_dir + '/cloud_inBody/*.txt'))

pcdfile_num = 10
pcd_number = len(pcd_files)
merge_num = int(pcd_number / pcdfile_num)
last_num = pcd_number - merge_num * pcdfile_num
print('waiting for merge ', last_num)
if (last_num / pcdfile_num * 1.0) > 0.7:
    merge_num += 1
print('total merge number ', merge_num)
i = 0
while i < merge_num:
    file_id = i * pcdfile_num
    pcd_path = pcd_files[file_id]
    combine_pcd = o3d.io.read_point_cloud(pcd_path)
    combine_numpy_pcd = np.asarray(combine_pcd.points)
    pose_path = pose_files[file_id]
    pose_file = open(pose_path, 'r', encoding='utf-8')
    pose1 = pose_file.readlines()
    T1 = np.zeros([4, 4])
    for k in range(3):
        T1[k] = [float(r) for r in pose1[k].split()]
    T1[3, 3] = 1
    R_1_W = T1[:3, :3].T
    t_1_W = -1 * T1[:3, :3].T @ T1[:3,3]

    for j in range(1, pcdfile_num, 1):
        if (file_id+j) > pcd_number:
            break
        pcd_path = pcd_files[file_id+j]
        pcd = o3d.io.read_point_cloud(pcd_path)
        numpy_pcd = np.asarray(pcd.points)
        pose_path = pose_files[file_id+j]
        pose_file = open(pose_path, 'r', encoding='utf-8')
        pose2 = pose_file.readlines()
        T2 = np.zeros([4, 4])
        for k in range(3):
            T2[k] = [float(r) for r in pose2[k].split()]
        T2[3, 3] = 1
        R_1_2 = np.dot(R_1_W, T1[:3,:3])
        t_1_2 = np.dot(R_1_W, T1[:3,3])+t_1_W
        new_numpy_pcd = np.dot(R_1_2, numpy_pcd.T).T + t_1_2
        combine_numpy_pcd = np.concatenate((combine_numpy_pcd, new_numpy_pcd))

    print("combine_numpy_pcd shape ", combine_numpy_pcd.shape)
    visual_pcd = o3d.geometry.PointCloud()
    visual_pcd.points = o3d.utility.Vector3dVector(combine_numpy_pcd)

    new_pcd_path = pcd_path.replace('cloud_inBody', 'cloud_inBody_mergeFrame')
    o3d.io.write_point_cloud(new_pcd_path, visual_pcd)
    new_pose_path = pose_path.replace('cloud_inBody', 'cloud_inBody_mergeFrame')
    shutil.copy2(pose_path, new_pose_path)

    i += 1
