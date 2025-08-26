from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from pyquaternion import Quaternion

parser = argparse.ArgumentParser(description='Process dataset parameters.')

parser.add_argument('--data_path', type=str, required=True,
                    help='Path to the dataset (mandatory)')
parser.add_argument('--dataset_version', type=str, default='test',
                    help='Version of the dataset (default: test)')
parser.add_argument('--camera_name', type=str, default='CAM_FRONT',
                    help='Name of the camera (default: CAM_FRONT)')

args = parser.parse_args()

data_path = args.data_path
dataset_version = args.dataset_version
camera_name = args.camera_name

# Init nuScenes
nusc = NuScenes(version='v1.0-' + dataset_version, dataroot=data_path, verbose=True)

# Pick a sample
sample = nusc.sample[0]  # use appropriate sample index

# Select tokens for lidar and camera
lidar_token = sample['data']['LIDAR_TOP']
camera_token = sample['data'][camera_name]

# Load lidar point cloud
lidar_data = nusc.get('sample_data', lidar_token)
pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data['filename']))

# Load camera data
cam_data = nusc.get('sample_data', camera_token)
im = plt.imread(os.path.join(nusc.dataroot, cam_data['filename']))

# def build_transformation_matrix(quat_wxyz, translation_matrix):
#     rotation_matrix = Quaternion(quat_wxyz).rotation_matrix
#     transition_matrix = np.eye(4)
#     transition_matrix[:3, :3] = rotation_matrix
#     transition_matrix[:3, 3] = translation_matrix

#     return transition_matrix

# def map_pointcloud_to_image(pc, cam_token, lidar_token, nusc, im, min_dist=1.0):
#     cam = nusc.get('sample_data', cam_token)
#     pointsensor = nusc.get('sample_data', lidar_token)
#     cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
#     poserecord = nusc.get('ego_pose', cam['ego_pose_token'])

#     # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
#     cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
#     lidar_to_ego_mat = build_transformation_matrix(cs_record['rotation'], cs_record['translation'])

#     # Second step: transform from ego to the global frame.
#     poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
#     lidar_ego_to_global_mat = build_transformation_matrix(poserecord['rotation'], poserecord['translation'])

#     # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
#     poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
#     camera_ego_to_global_mat = build_transformation_matrix(poserecord['rotation'], poserecord['translation'])

#     # Fourth step: transform from ego into the camera.
#     cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
#     camera_to_ego_mat = build_transformation_matrix(cs_record['rotation'], cs_record['translation'])

#     lidar_to_camera_mat = np.linalg.inv(camera_to_ego_mat) @ np.linalg.inv(camera_ego_to_global_mat) @ lidar_ego_to_global_mat @ lidar_to_ego_mat
#     hom_points = np.vstack((pc.points[:3, :], np.ones((1, pc.points.shape[1]))))
#     pc_cam_points = lidar_to_camera_mat @ hom_points
#     pc = LidarPointCloud(pc_cam_points)

#     # Project onto image
#     cam_intrinsic = np.array(cs_record['camera_intrinsic'])
#     points = view_points(pc.points[:3, :], cam_intrinsic, normalize=True)

#     depths = pc.points[2, :]
#     mask = np.ones(depths.shape[0], dtype=bool)
#     mask = np.logical_and(mask, depths > min_dist)
#     mask = np.logical_and(mask, points[0, :] > 1)
#     mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
#     mask = np.logical_and(mask, points[1, :] > 1)
#     mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)

#     return points[:, mask]

# # 1. Use the function
# xy_points = map_pointcloud_to_image(pc, camera_token, lidar_token, nusc, im)

# plt.imshow(im)
# plt.scatter(xy_points[0, :], xy_points[1, :], s=1, c='r')
# plt.axis('off')
# plt.show()

# 2. Use library
nusc.render_pointcloud_in_image(
    sample['token'],
    pointsensor_channel='LIDAR_TOP',
    camera_channel=camera_name,
    render_intensity=True)