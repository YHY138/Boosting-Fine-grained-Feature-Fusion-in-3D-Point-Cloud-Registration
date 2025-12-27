'''
This code is designed to downsample PCD files containing label information.
'''
import os, sys, glob, time
import torch
import numpy as np
import open3d as o3d
from pypcd import pypcd


def downsample_array(array, labels, ratio):
    unique_labels = np.unique(labels)  
    sampled_array = np.empty((0, array.shape[1])) 
    sampled_labels = np.empty(0, dtype=labels.dtype) 

    for label in unique_labels:
        label_array = array[labels == label] 
        label_sampled_array = label_array[:int(len(label_array) * ratio)]
        sampled_array = np.concatenate((sampled_array, label_sampled_array)) 
        sampled_labels = np.concatenate((sampled_labels, np.full(len(label_sampled_array), label, dtype=labels.dtype))) 

    return sampled_array, sampled_labels


if __name__ == '__main__':
    pcd_dir = 'path_to_your_pcd_directory'  # Replace with your PCD directory path
    dekewed_pcd = glob.glob(pcd_dir + '/inL_labelled/*.pcd')

    for i in range(len(dekewed_pcd)):
        print("Processing ", dekewed_pcd[i])
        pth_file_name = dekewed_pcd[i].replace('.pcd', '.pth')
        pth_file_name = pth_file_name.replace('inL_labelled', 'inL_labelled_Sample_4kpts')
        label_file_name = pth_file_name.replace('.pth', '_label.txt')
        cloud1 = pypcd.PointCloud.from_path(dekewed_pcd[i]).pc_data 
        cloud1 = np.column_stack((cloud1['x'], cloud1['y'], cloud1['z'], cloud1['label']))

        xyz_arr = cloud1[:, :3]
        label_arr = cloud1[:, 3]
        sample_xyz, sample_label = downsample_array(xyz_arr, label_arr, 0.46)
        tensor = torch.from_numpy(sample_xyz)
        tensor = tensor.float() 
        tensor = tensor.numpy()
        tensor = tensor.astype(np.float32) 
        torch.save(tensor, pth_file_name)
        label_file = open(label_file_name, 'w')
        for label in sample_label:
            label_file.write(str(int(label)) + ' ')

