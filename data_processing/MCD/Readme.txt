Download the dataset from the official MCD website, which contains bag files and the pose_inW.csv file.
And use follow scripts for data processing.

python get_mcdPCD&Pose.py # get pointcloud and pose file

python combine_multiFrame.py # merge multiple pointcloud to enhance structure information, otherwise original LiDAR pointcloud is sparse.

python pcd_to_samplepth.py # sample pointcloud to avoid GPU out of memory during training

Subsequently, the Create code is utilized to generate the metadata required for training.

TODO: update the semantic data processing
