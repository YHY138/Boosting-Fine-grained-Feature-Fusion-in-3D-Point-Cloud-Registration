
"""
We use this script to calculate the overlap ratios for all the train/test fragment pairs
"""
import os, sys, glob, re
import open3d as o3d
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def get_overlap_ratio(source, target, threshold=0.0375):
    """
    We compute overlap ratio from source point cloud to target point cloud
    """
    pcd_tree = o3d.geometry.KDTreeFlann(target)

    match_count = 0
    for i, point in enumerate(source.points):
        [count, _, _] = pcd_tree.search_radius_vector_3d(point, threshold)
        if (count != 0):
            match_count += 1

    overlap_ratio = match_count / len(source.points)
    return overlap_ratio

def cal_overlap_per_scene(c_folder):
    base_dir = os.path.join(c_folder, 'cloud_inBody')
    fragments = sorted(glob.glob(base_dir + '/*.pcd'), key=natural_key)
    n_fragments = len(fragments)

    with open(f'{c_folder}/overlaps_ours.txt', 'w') as f:
        for i in tqdm(range(n_fragments - 1)):
            for j in range(i + 1, n_fragments):
                path1, path2 = fragments[i], fragments[j]

                # load, downsample and transform
                pcd1 = o3d.io.read_point_cloud(path1)
                pcd2 = o3d.io.read_point_cloud(path2)
                pcd1 = pcd1.voxel_down_sample(0.01)
                pcd2 = pcd2.voxel_down_sample(0.01)

                # calculate overlap
                c_overlap = get_overlap_ratio(pcd1, pcd2)
                print(c_overlap)
                f.write(f'{i},{j},{c_overlap:.4f}\n')
        f.close()

def cal_overlap_twoPCD(pcd1, pcd2, threshold):
    # calculate overlap
    c_overlap = get_overlap_ratio(pcd1, pcd2, threshold)
    return c_overlap

if __name__ == '__main__':
    exported_dir = 'tuhh_night_09_os1_64_exported_pcds'
    scenes = sorted(glob.glob(exported_dir))

    p = mp.Pool(processes=mp.cpu_count())
    p.map(cal_overlap_per_scene, scenes)
    p.close()
    p.join()
