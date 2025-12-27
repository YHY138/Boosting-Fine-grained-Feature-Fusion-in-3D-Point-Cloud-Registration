# Boosting Fine-grained Feature Fusion in 3D Point Cloud Registration
Source code for the paper: Boosting Fine-grained Feature Fusion in 3D Point Cloud Registration. Manuscript ID: IEEE LATAM Submission ID: 10157 

---

##  Project Structure
├── backbone/ # Backbone network implementation  
├── conf/ #  Training configuration  
├── cvhelpers/ #  Visulization tool  
├── data_loaders/ #   Data load tool
├── evo/ #   Evaluating RMSE
├── data_processing/ # 3DMatch and MCD Dataset Processing   
├── kernels/   Kernels dispositions
├── model/ # Model architecture  
├── utils # Variable processing tool  
├── README.md # Project documentation 
├── requirements.txt # Dependency list  
├── rr_loop.sh # Validation script for Registration Recall   
├── save_result_demo.py # Saving registration results  
├── test.py #  Registration Recall calculation
├── train.py # Training script   
├── trainer.py #  Trainer and optimizer

---

## Datasets
- [3DMatch](https://share.phys.ethz.ch/~gseg/Predator/data.zip)
- [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)
- [MCD](https://mcdviral.github.io/)

---

## Requirement
- Python 3.9
- Pytorch 1.12.1
- CUDA 11.6
- MinkowskiEngine 0.5.4
- PyTorch3D 0.7.5

```bash
pip install -r requirements.txt  
```
---

## Training
```bash
python train.py --config conf/dataset_name.yaml
```

---

## Validation for Registration Recall
```bash
bash ./rr_loop.sh
```

---

## Validation for RMSE
```bash
python save_result_demo.py 
python evo/convert_evo.py
# then use evo_ape for RMSE evaluation. evo tutorial https://github.com/MichaelGrupp/evo
```

---

## Save Colored Point Clouds
```bash
python save_colorP3D.py
```

---