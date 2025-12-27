'''
This code is designed to obtain the pose of each point cloud frame as well as 
its coordinates in the Body coordinate system, based on the real ground truth (GT) and bag files.
This code return csvPose
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

# Link to rosbag
rosbag_file = '/data/MCD_dataset/4-ntu_night_08/ntu_night_08_os1_128.bag'

# Folder to export the pointclouds to
exported_dir = '/data/MCD_dataset/processed_data/ntu_night_08_exported_pcds_forTrain'

# Ground truth spline log
spline_log = '/data/MCD_dataset/4-ntu_night_08/spline.csv'
# Relative pose between Lidar and Body coordinate system
T_B_L = np.array([[0.9999346552051229, 0.003477624535771754, -0.010889970036688295, -0.060649229060416594],
                  [0.003587143302461965, -0.9999430279821171, 0.010053516443599904, -0.012837544242408117],
                  [-0.010854387257665576, -0.01009192338171122, -0.999890161647627, -0.020492606896077407],
                  [ 0.000000000000000000,  0.000000000000000000,  0.000000000000000000,  1.000000000000000000]])

# Minimum range
min_range = 0.75

# Load the ground truth

# Read the spline
log = open(spline_log, 'r', encoding="gbk")

# Extract some settings in the header
log_header = log.readline()
log.close()

# Read the dt from header
dt = float(log_header.split(sep=',')[0].replace('Dt: ', ''))
# Read the spline order
order = int(log_header.split(sep=',')[1].replace('Order: ', ''))
# Read the number of knots
knots = int(log_header.split(sep=',')[2].replace('Knots: ', ''))
# Read the start time in header
start_time = float(log_header.split(sep=',')[3].replace('MinTime: ', ''))
# Calculate the end time
final_time = start_time + dt*(knots - order + 1)

# Read the spline log in text
knots = np.loadtxt(spline_log, delimiter=',', skiprows=1)
spline = Ceva(order, dt, start_time, spline_log)

# Create folders
os.makedirs(exported_dir + '/cloud_inBody', exist_ok=True)

# Read the pointclouds from rosbag and deskew them

# Extract the rotation and translation extrinsics for convinience
R_B_L = T_B_L[:3, :3]
t_B_L = T_B_L[:3, 3]

# Create interface with rosbag
bag = rosbag.Bag(rosbag_file)

# Cloud index
cloud_idx = -1

# Iterate through the message
for topic, msg, t in bag.read_messages():

    if topic == '/os_cloud_node/points':
        cloud_idx += 1

        # Ouster lidar timestamp is incident with the last point
        sweeptime = 0.1
        timestamp = msg.header.stamp

        # Make sure the pointcloud is in the valid time period, adding some padding for certainty
        if timestamp.to_sec() < spline.minTime() + sweeptime + 10e-3 or timestamp.to_sec() > spline.maxTime() - sweeptime - 10e-3:
            continue

        # Convert the msg to np array
        pc_in = np.squeeze(np.reshape(pypcd.PointCloud.from_msg(msg).pc_data, (msg.height * msg.width, 1)))

        # Adjust the sweep time to be more precise
        sweeptime = (max(pc_in['t']) - min(pc_in['t'])) / 1.0e9

        # Time stamp at the beginning
        timestamp_beginning = timestamp - rospy.Duration(sweeptime)

        # Obtain the pose at the beginning of the scan
        pose_ts = spline.getPose(timestamp_beginning.to_sec())[0]  # t, x, y, z, qx, qy, qz, qw
        R_W_Bs = Rotation.from_quat(pose_ts[4:8]).as_matrix()
        t_W_Bs = pose_ts[1:4].T

        # Remove the zero points
        nonzero_idx = (pc_in['range'] / 1000.0 > min_range).nonzero()[0]
        pc_in = pc_in[nonzero_idx]

        # Extract the xyz parts
        pc_xyz_inL = np.array([list(p) for p in pc_in[['x', 'y', 'z']]])

        # Extract the intensity
        intensity = pc_in['intensity'].reshape((len(pc_in['intensity']), 1))

        # Extract the point timestamp relative to the beginning
        tr = pc_in['t']

        # Absolute time of the points
        th = timestamp_beginning.to_sec()
        ta = th + tr / 1.0e9

        # Transform pointcloud from lidar to body frame
        pc_xyz_inB = np.dot(R_B_L, pc_xyz_inL.T).T + t_B_L # 这里算的是Lidar看到的三维点变换到Body坐标系上的坐标        # Put the intensity back
        pc_xyz_inB = np.concatenate([pc_xyz_inB, intensity], axis=1) # 这里算的是PCD在世界坐标系上的坐标值。

        # Save the original distorted pointcloud
        pc_inW_distorted_filename = exported_dir + '/cloud_inBody/cloud' \
                                    + '_' + str(msg.header.seq).zfill(4) \
                                    + '_' + f'{timestamp_beginning.secs}_{timestamp_beginning.nsecs}.pcd'
        pypcd.save_point_cloud_bin_compressed(
            pypcd.PointCloud.from_array(np.array(list(map(tuple, pc_xyz_inB)), dtype=[('x', '<f4'),
                                                                                            ('y', '<f4'),
                                                                                            ('z', '<f4'),
                                                                                            ('intensity', '<f4'),
                                                                                            ])),
            pc_inW_distorted_filename)
        cloud_pose_filename = pc_inW_distorted_filename + '.txt'
        pose_file = open(cloud_pose_filename, 'w')
        last_str = '{:e} {:e} {:e} {:e} \n'.format(0, 0, 0, 1)
        for i in range(3):
            temp = ['{:e}'.format(x) for x in R_W_Bs[i, :]]
            temp.append('{:e}'.format(t_W_Bs[i]))
            pose_file.write(' '.join(temp) + '\n')
        pose_file.write(last_str)

# Close the rosbag
bag.close()