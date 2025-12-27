"""Precomputes the overlap regions for 3DMatch dataset,
used for computing the losses in RegTR.
"""
import argparse
import os
import pickle
import sys
sys.path.append(os.getcwd())

import h5py
import numpy as np
import torch
from tqdm import tqdm

from utils.pointcloud import compute_overlap
from utils.se3_numpy import se3_transform, se3_init

parser = argparse.ArgumentParser()
# General
parser.add_argument('--base_dir', type=str, default='/home/dell/anaconda3/envs/yhy/MCD_datasets/TUHH/',
                    help='Path to MCD raw data (Predator format)')
parser.add_argument('--overlap_radius', type=float, default=1,
                    help='Overlap region will be sampled to this voxel size')
opt = parser.parse_args()


def process(phase):
    opt.base_dir = '/home/dell/anaconda3/envs/yhy/MCD_datasets/TUHH/ntu_night_08_exported_pcds_forTrain'
    with open(f'/home/dell/anaconda3/envs/yhy/MCD_datasets/TUHH/ntu_night_08_exported_pcds_forTrain/ntu_night_08_mergeMiniGap_{phase}.pkl', 'rb') as fid:
    # with open(f'/home/ubuntu/anaconda3/envs/regtr/MCD_dataset/TUHH_dataset/tuhh_night_09_os1_64-001_mini.pkl', 'rb') as fid:
        infos = pickle.load(fid)

    out_file = os.path.join(opt.base_dir, f'{phase}_pairs-overlapmask.h5')
    print(f'Processing {phase}, output: {out_file}...')
    h5_fid = h5py.File(out_file, 'w')

    num_pairs = len(infos['src'])
    for item in tqdm(range(num_pairs)):
        src_path = infos['src'][item]
        tgt_path = infos['tgt'][item]
        # print(infos['rot'][item])
        # print(infos['trans'][item].shape)
        pose = se3_init(infos['rot'][item], infos['trans'][item])  # transforms src to tgt

        src_xyz = torch.load(os.path.join(opt.base_dir, src_path))
        tgt_xyz = torch.load(os.path.join(opt.base_dir, tgt_path))

        src_mask, tgt_mask, src_tgt_corr = compute_overlap(
            se3_transform(pose, src_xyz),
            tgt_xyz,
            opt.overlap_radius,
        )

        h5_fid.create_dataset(f'/pair_{item:06d}/src_mask', data=src_mask)
        h5_fid.create_dataset(f'/pair_{item:06d}/tgt_mask', data=tgt_mask)
        h5_fid.create_dataset(f'/pair_{item:06d}/src_tgt_corr', data=src_tgt_corr)


if __name__ == '__main__':
    process('train')
    process('val')