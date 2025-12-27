'''
This is designed to construct a pkl dataset compatible with Regtr testing using the SemanticMCD dataset, 
with the requirement that Label, pose and pth files must be located in the same directory.
'''

import os, sys, glob, time
import numpy as np

import open3d as o3d
import pickle

import torch

from compute_MCDoverlap import cal_overlap_twoPCD

# Folder to export the pointclouds to
exported_dir = 'your_exported_dir'
pth_files = sorted(glob.glob(exported_dir + '/inL_labelled_Sample_4kpts/*.pth'))
pose_files = sorted(glob.glob(exported_dir + '/inL_labelled_Sample_4kpts/*_pose.txt'))
label_files = sorted(glob.glob(exported_dir + '/inL_labelled_Sample_4kpts/*_label.txt'))

gtlog_files = open(exported_dir + '/inL_labelled_Sample_4kpts/'+'gt.log','w')
last_str = '{:e} {:e} {:e} {:e} \n'.format(0, 0, 0, 1)
src, tgt, rot, trans, overlap, src_labels, tgt_labels = [], [], [], [], [], [], []
pcd_number = len(pth_files)
total_num = 0 
train_ratio = 1
for i in range(pcd_number):
    for j in range(i+1, i+2):
        if not j < pcd_number:
            continue
        print("Processing\n", pth_files[i])
        print(pth_files[j])
        # 读取pcd和pose文件内容
        src_files, tgt_files = pth_files[i], pth_files[j]
        pose_path1, pose_path2 = pose_files[i], pose_files[j]
        label_path1, label_path2 = label_files[i], label_files[j]
        pcd_numpy1 = torch.load(src_files)
        pcd_numpy2 = torch.load(tgt_files)
        pcd1 = o3d.geometry.PointCloud()
        print(pcd_numpy1.shape)
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

        overlap_ratio = cal_overlap_twoPCD(pc_o3d1, pc_o3d2, 1)
        overlap.append(overlap_ratio)
        src.append(src_files)
        tgt.append(tgt_files)
        rot.append(R_2_1)
        trans.append(t_2_1.reshape(3,1))

        label_f1 = open(label_path1, 'r')
        src_label_str = label_f1.readline()
        label_str = src_label_str.split(' ')
        label1 = []
        for l in label_str:
            try:
                label1.append(int(l))
            except:
                continue
        label1 = np.array(label1)
        print(label1.shape)
        src_labels.append(label1)

        label_f2 = open(label_path2, 'r')
        tgt_label_str = label_f2.readline()
        label_str = tgt_label_str.split(' ')
        label2 = []
        for l in label_str:
            try:
                label2.append(int(l))
            except:
                continue
        label2 = np.array(label2)
        tgt_labels.append(label2)

        str_ids = str(i) + ' ' + str(j) + '\n'
        gtlog_files.write(str_ids)
        for p in range(3):
            temp = ['{:e}'.format(x) for x in R_2_1[p, :]]
            temp.append(str(t_2_1[p]))
            gtlog_files.write(' '.join(temp) + '\n')
        gtlog_files.write(last_str)

        total_num += 1

kitti_pkl = {}
kitti_pkl['src'] = np.array(src[:int(total_num*train_ratio)])
kitti_pkl['tgt'] = np.array(tgt[:int(total_num*train_ratio)])
kitti_pkl['overlap'] = np.array(overlap[:int(total_num*train_ratio)])
kitti_pkl['rot'] = np.array(rot[:int(total_num*train_ratio)])
kitti_pkl['trans'] = np.array(trans[:int(total_num*train_ratio)])
kitti_pkl['src_labels'] = np.array(src_labels[:int(total_num*train_ratio)], dtype=object)
kitti_pkl['tgt_labels'] = np.array(tgt_labels[:int(total_num*train_ratio)], dtype=object)

print(kitti_pkl.keys())
print(kitti_pkl['overlap'].shape)#<class 'numpy.ndarray'>(n,)
print(type(kitti_pkl['overlap']))
print(kitti_pkl['src'].shape)
print(kitti_pkl['rot'].shape)# <class 'numpy.ndarray'> (n, 3,3)
print(type(kitti_pkl['rot']))
print(kitti_pkl['trans'].shape)# <class 'numpy.ndarray'> （n, 3, 1)

file = open(os.path.join(exported_dir, 'tuhh_day_03_Labelled_4kpts_test_for_tuhhnight08train.pkl'), 'wb')
pickle.dump(kitti_pkl, file)
file.close()


