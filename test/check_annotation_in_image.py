from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
import numpy as np
import matplotlib.pyplot as plt
import os
from pyquaternion import Quaternion
import argparse

parser = argparse.ArgumentParser(description='Process dataset parameters.')

parser.add_argument('--data_path', type=str, required=True,
                    help='Path to the dataset (mandatory)')
parser.add_argument('--dataset_version', type=str, default='test',
                    help='Version of the dataset (default: test)')
parser.add_argument('--sample_num', type=int, default=0,
                    help='Number of the sample (default: 0)')
parser.add_argument('--mode', type=int, default=2,
                    help='Test mode - camera / lidar / annotation (default: 2)')

args = parser.parse_args()

data_path = args.data_path
dataset_version = args.dataset_version
sample_num = args.sample_num
mode = args.mode

# Init nuScenes
nusc = NuScenes(version='v1.0-' + dataset_version, dataroot=data_path, verbose=True)

# Pick a sample
sample = nusc.sample[sample_num]  # use appropriate sample index
lidar_token = sample['data']['LIDAR_TOP']

if mode == 0:
    # camera
    for sensor_name in sample['data']:
        if 'CAM' in sensor_name:
            camera_token = sample['data'][sensor_name]

            # Load lidar point cloud
            lidar_data = nusc.get('sample_data', lidar_token)
            pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data['filename']))

            # Load camera data
            cam_data = nusc.get('sample_data', camera_token)
            im = plt.imread(os.path.join(nusc.dataroot, cam_data['filename']))

            nusc.render_sample_data(cam_data['token'])
elif mode == 1:
    # lidar
    nusc.render_sample_data(lidar_token, axes_limit=60)
elif mode == 2:
    # annotation
    nusc.render_annotation(sample['anns'][0], margin=40)
    plt.show()
else:
    print("Undefined mode")
