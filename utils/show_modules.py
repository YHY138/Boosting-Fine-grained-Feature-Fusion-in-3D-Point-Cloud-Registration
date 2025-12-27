"""Barebones code demonstrating REGTR's registration. We provide 2 demo
instances for each dataset
Simply download the pretrained weights from the project webpage, then run:
    python demo.py EXAMPLE_IDX
where EXAMPLE_IDX can be a number between 0-5 (defined at line 25)
The registration results will be shown in a 3D visualizer.
"""
import argparse
from pathlib import Path
import torch
from easydict import EasyDict
from models.regtr import RegTR
from utils.misc import load_config


_examples = [
    # 3DMatch examples
    # 0
    ('../trained_models/3dmatch/ckpt/model-best.pth',
     '../data/indoor/test/7-scenes-redkitchen/cloud_bin_0.pth',
     '../data/indoor/test/7-scenes-redkitchen/cloud_bin_5.pth'),
    # 1
    ('../trained_models/3dmatch/ckpt/model-best.pth',
     '../data/indoor/test/sun3d-hotel_umd-maryland_hotel3/cloud_bin_8.pth',
     '../data/indoor/test/sun3d-hotel_umd-maryland_hotel3/cloud_bin_15.pth'),
    # 2
    ('../trained_models/3dmatch/ckpt/model-best.pth',
     '../data/indoor/test/sun3d-home_at-home_at_scan1_2013_jan_1/cloud_bin_38.pth',
     '../data/indoor/test/sun3d-home_at-home_at_scan1_2013_jan_1/cloud_bin_41.pth'),
    # ModelNet examples
    # 3
    ('../trained_models/modelnet/ckpt/model-best.pth',
     '../data/modelnet_demo_data/modelnet_test_2_0.ply',
     '../data/modelnet_demo_data/modelnet_test_2_1.ply'),
    # 4
    ('../trained_models/modelnet/ckpt/model-best.pth',
     '../data/modelnet_demo_data/modelnet_test_630_0.ply',
     '../data/modelnet_demo_data/modelnet_test_630_1.ply'),
]

parser = argparse.ArgumentParser()
parser.add_argument('--example', type=int, default=0,
                    help=f'Example pair to run (between 0 and {len(_examples) - 1})')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Controls viusalization of keypoints outside overlap region.')
opt = parser.parse_args()


def main():
    # Retrieves the model and point cloud paths
    ckpt_path, src_path, tgt_path = _examples[opt.example]

    # Load configuration file
    cfg = EasyDict(load_config(Path(ckpt_path).parents[1] / 'config.yaml'))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Instantiate model and restore weights
    model = RegTR(cfg).to(device)
    state = torch.load(ckpt_path)
    model.load_state_dict(state['state_dict'])

    print(model._modules)



if __name__ == '__main__':
    main()