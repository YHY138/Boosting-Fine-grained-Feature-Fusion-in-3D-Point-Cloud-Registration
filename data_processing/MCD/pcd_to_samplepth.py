'''
这个代码是为了将pcd格式的文件转成pth，同时对pcd中的点降采样，以达到降低文件大小的目的，不然训练显存完全不够用
'''
import os, sys, glob, time
import torch
import numpy as np
import open3d as o3d

exported_dir = 'exported_dir'  # Replace with your PCD directory path
pcd_files = sorted(glob.glob(exported_dir + '/cloud_inBody_mergeFrame_withGap/*.pcd'))

for i in range(len(pcd_files)):
    print("Processing ", pcd_files[i])
    pth_file = pcd_files[i].replace('.pcd', '.pth')
    pth_file = pth_file.replace('cloud_inBody_mergeFrame_withGap', 'cloud_inBody_mergeFrame_withGapSample')
    cloud = o3d.io.read_point_cloud(pcd_files[i])
    sample_cloud = cloud.uniform_down_sample(350) 

    pt_num = np.asarray(sample_cloud.points)
    print(pt_num.shape)
    points = np.asarray(sample_cloud.points)
    tensor = torch.from_numpy(points)
    tensor = tensor.float() 
    tensor = tensor.numpy() 
    tensor = tensor.astype(np.float32) 
    torch.save(tensor, pth_file)


