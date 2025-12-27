'''
This code converts the MCD dataset into a PKL file format consistent with 3DMatch, 
based on the PCD files corresponding to each pose,
and requires computing the relative pose relationship between two PCD files.
'''
import os, sys, glob, time
import numpy as np
import scipy
from scipy.spatial.transform import Rotation

import rospy, rosbag
# python wrapper for spline
from ceva import Ceva
# pcd interface
from pypcd import pypcd
import open3d as o3d
from tqdm import tqdm
import pickle

import torch

# 引入predator里面的重迭率计算公式，但好像没啥用我记得，regtr训练时好像不用这个重迭率
from compute_MCDoverlap import cal_overlap_twoPCD

# Folder to export the pointclouds to
exported_dir = '/home/dell/anaconda3/envs/yhy/MCD_datasets/TUHH/tuhh_night_09_os1_64_exported_pcds'
# 获得所有pcd文件以及对应的位姿txt文件
# pcd_files = sorted(glob.glob(exported_dir + '/cloud_inBody/*.pcd'))
pth_files = sorted(glob.glob(exported_dir + '/cloud_inBody_mergeFrameSample/*.pth'))
# pose_files = sorted(glob.glob(exported_dir + '/cloud_inBody/*.txt'))
pose_files = sorted(glob.glob(exported_dir + '/cloud_inBody_mergeFrameSample/*.txt'))
# print(pcd_files[:10])
# print(pose_files[:10])
# src保存的是文件路径和名称，tgt也是。rot保存的是np.array类型的旋转矩阵，维度是(n,3,3),trans则是平移，维度同rot。overlap是算出来的重叠率，也是np.array，维度是（n，）
# mcd_pkl = {'src': [], 'tgt':[], 'rot': None, 'trans':None, 'overlap':[]}
# 下面这些src、tgt都是训练样本
src, tgt, rot, trans, overlap = [], [], [], [], []
pcd_number = len(pth_files)
total_num = 0 # 统计总共有多少组数据
train_ratio = 0.8
# 先只做100组数据，看能否满足后续处理的操作
for i in range(pcd_number):

    for j in range(i+1, pcd_number):
        if i == j:
            continue
        print("当前正在处理\n", pth_files[i])
        print(pth_files[j])
        # 读取pcd和pose文件内容
        src_files, tgt_files = pth_files[i], pth_files[j]
        pose_path1, pose_path2 = pose_files[i], pose_files[j]
        # 加载PCD点云和位姿矩阵
        pcd_numpy1 = torch.load(src_files)
        pcd_numpy2 = torch.load(tgt_files)
        # print(type(pcd1))
        # pcd1 = o3d.io.read_point_cloud(src_files)
        # pcd2 = o3d.io.read_point_cloud(tgt_files)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcd_numpy1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcd_numpy2)
        # 对点云进行降采样
        # pcd1 = pcd1.voxel_down_sample(0.01)
        # pcd2 = pcd2.voxel_down_sample(0.01)
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
        # 根据获得到的点和重叠率以及位姿，创建与3DMatch类似的pkl文件
        # pkl里面的src关键词包含的不是三维点的坐标，而是对应的srcPCD文件名
        # src_files = pcd_path1.replace('.pcd','.pth')
        # src_files = src_files.replace('cloud_inBody','cloud_inBody_minisampler4')
        # tgt_files = pcd_path2.replace('.pcd','.pth')
        # tgt_files = tgt_files.replace('cloud_inBody','cloud_inBody_minisampler4')
        print("@@@@@@@@@@@@ 当前计算的id号", i, ' ', j)
        print("@####### 对应的pth文件 \n", src_files)
        print(tgt_files)

        # 计算世界坐标系到tgt点云的位姿变换矩阵，即T2的逆
        R_2_W = T2[:3, :3].T
        t_2_W = -1 * T2[:3, :3].T @ T2[:3,3]

        # 显示一下基于相对位姿变化的点云，看是否算对了两个点云各自到世界坐标系的位姿变换矩阵。算对了
        np_pcd1 = np.asarray(pcd1.points)
        np_pcd2 = np.asarray(pcd2.points)
        # 经过这一条运算后，nppcd1变量已经处于世界坐标系下了
        np_pcd1 = np.dot(T1[:3,:3], np_pcd1.T).T + T1[:3,3]
        np_pcd2 = np.dot(T2[:3,:3], np_pcd2.T).T + T2[:3,3]
        pc_o3d1 = o3d.geometry.PointCloud()
        pc_o3d1.points = o3d.utility.Vector3dVector(np_pcd1)
        pc_o3d1.paint_uniform_color([0, 1, 0])
        pc_o3d2 = o3d.geometry.PointCloud()
        pc_o3d2.points = o3d.utility.Vector3dVector(np_pcd2)
        pc_o3d2.paint_uniform_color([0, 0, 1])
        # o3d.visualization.draw_geometries([pc_o3d1, pc_o3d2])

        # 计算两个点云之间的相对位姿变换
        R_2_1 = np.dot(R_2_W, T1[:3,:3])
        t_2_1 = np.dot(R_2_W, T1[:3,3])+t_2_W

        # print(t_2_1.shape)
        ## 先将pcd1变换到世界坐标系
        # np_pcd1 = np.asarray(pcd1.points)
        # np_pcd1 = np.dot(T1[:3, :3], np_pcd1.T).T + T1[:3, 3]
        ## 再将其从世界坐标系投影到pcd2所在坐标系
        ## inpcd2_np_pcd1 = np.dot(R_2_W, np_pcd1.T).T + t_2_W
        # inpcd2_np_pcd1 = np.dot(R_2_1, pcd_numpy1.T).T + t_2_1
        # pcd1_in_pcd2 = o3d.geometry.PointCloud()
        # pcd1_in_pcd2.points = o3d.utility.Vector3dVector(inpcd2_np_pcd1)
        # pcd1_in_pcd2.paint_uniform_color([0, 1, 0])
        # pcd2.paint_uniform_color([0, 0, 1])
        # # 显示
        # o3d.visualization.draw_geometries([pcd2, pcd1_in_pcd2])

        # 计算两个PCD点云之间的重迭率，这次将两个点投影到世界坐标系后再计算重叠率，以防止重叠率过低
        overlap_ratio = cal_overlap_twoPCD(pc_o3d1, pc_o3d2, 0.5)
        print("########## 两组pcd之间的重叠率为：", overlap_ratio)
        # 只有当overlapratio大于0.2的时候才保留结果到pkl文件中
        overlap.append(overlap_ratio)
        src.append(src_files)
        tgt.append(tgt_files)## 保存相对位姿变换结果
        rot.append(R_2_1)
        trans.append(t_2_1.reshape(3,1)) # 保存平移的时候，需要平移向量的维度是（3，1）。上面所有处理的过程中，t21的维度都是(3,)
        total_num += 1

# 将所有list保存到字典中，并构建成pkl文件
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
# 检查以下格式
print(mcd_pkl.keys())
print(mcd_pkl['overlap'].shape)#<class 'numpy.ndarray'>(n,)
print(type(mcd_pkl['overlap']))
print(mcd_pkl['rot'].shape)# <class 'numpy.ndarray'> (n, 3,3)
print(type(mcd_pkl['rot']))
print(mcd_pkl['trans'].shape)# <class 'numpy.ndarray'> （n, 3, 1)
# 开始划分train集和val集

# 生成对应的pkl文件
file = open('./tuhh_night_09_os1_64-001_mergeMini_train.pkl', 'wb')
pickle.dump(mcd_pkl, file)
file.close()

val_file = open('./tuhh_night_09_os1_64-001_mergeMini_val.pkl', 'wb')
pickle.dump(val_mcdpkl, val_file)
val_file.close()


