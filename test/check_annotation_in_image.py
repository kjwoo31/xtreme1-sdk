from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
import numpy as np
import matplotlib.pyplot as plt
import os
from pyquaternion import Quaternion

# Init nuScenes
nusc = NuScenes(version='v1.0-test', dataroot='/mnt/OT_DATASET/dataset/nuscenes/data/nuscenes', verbose=True)

# Pick a sample
sample = nusc.sample[0]  # use appropriate sample index

# Select tokens for lidar and camera
lidar_token = sample['data']['LIDAR_TOP']
camera_token = sample['data']['CAM_FRONT']

# Load lidar point cloud
lidar_data = nusc.get('sample_data', lidar_token)
pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data['filename']))

# Load camera data
cam_data = nusc.get('sample_data', camera_token)
im = plt.imread(os.path.join(nusc.dataroot, cam_data['filename']))

nusc.render_sample_data(cam_data['token'])
