'''
This code is designed to convert files in PCD format to PTH format, 
while simultaneously performing downsampling on the points in the PCD files to reduce the file size.
'''
import os, sys, glob, time
import torch
import numpy as np
import open3d as o3d
import shutil

exported_dir = '/path/to/your/MCD/dataset'
pcd_files = sorted(glob.glob(exported_dir + '/cloud_inBody_mergeFrame/*.pcd'))
pose_files = sorted(glob.glob(exported_dir + '/cloud_inBody_mergeFrame/*.txt'))

output_dir = os.path.join(exported_dir, 'cloud_inBody_mergeFrameSample')
os.makedirs(output_dir, exist_ok=True)

for i in range(len(pcd_files)):
    print("Processing ", pcd_files[i])
    pth_file = pcd_files[i].replace('.pcd', '.pth')
    pth_file = pth_file.replace('cloud_inBody_mergeFrame', 'cloud_inBody_mergeFrameSample')
    cloud = o3d.io.read_point_cloud(pcd_files[i])
    sample_cloud = cloud.uniform_down_sample(350) 

    pt_num = np.asarray(sample_cloud.points)
    points = np.asarray(sample_cloud.points)
    tensor = torch.from_numpy(points)
    tensor = tensor.float() 
    tensor = tensor.numpy() 
    tensor = tensor.astype(np.float32) 
    torch.save(tensor, pth_file)

    # save pose
    new_pose_file = pose_files[i].replace('.pcd','.pth')
    new_pose_file = new_pose_file.replace('cloud_inBody_mergeFrame', 'cloud_inBody_mergeFrameSample')
    shutil.copy2(pose_files[i], new_pose_file)


