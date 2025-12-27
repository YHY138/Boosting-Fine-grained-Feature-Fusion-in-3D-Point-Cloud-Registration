'''
这个代码是为了将pcd格式的文件转成pth，方便pytorch读取和使用
'''
import os, sys, glob, time
import shutil

exported_dir = '/home/dell/anaconda3/envs/yhy/MCD_datasets/TUHH/ntu_day_10_exported_pcds_forTest'
pose_files = sorted(glob.glob(exported_dir + '/cloud_inBody_mergeFrame_withGap/*.txt'))

for i in range(len(pose_files)):
    if "说明" in pose_files[i]:
        continue
    new_pose_file = pose_files[i].replace('.pcd','.pth')
    new_pose_file = new_pose_file.replace('cloud_inBody_mergeFrame_withGap', 'cloud_inBody_mergeFrame_withGapSample')
    shutil.copy2(pose_files[i], new_pose_file)
