from xtreme1.client import Client
import argparse
import os
import json

x1_client = Client(
    base_url='http://localhost:8190', 
    access_token='eyJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJCYXNpY0FJIiwiaWF0IjoxNzU0OTY1OTM1LCJzdWIiOiIyIn0.IfNp3s4GdC24hruNIvGymUgqNX0irdtQoWUxfQi3pYPCipmtVtCsL9iyOADhYrrXbUGv1ep09b5LRqq4nkjH1w'
)

parser = argparse.ArgumentParser(description='Process dataset parameters.')

parser.add_argument('--dataset_name', type=str, required=True,
                    help='Name of the dataset (mandatory)')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Path to output results (mandatory)')

args = parser.parse_args()

dataset_name = args.dataset_name
output_dir = args.output_dir
dataset_dir = os.path.join(output_dir, dataset_name)

# Query a list of datasets
dataset_list, total = x1_client.query_dataset()
dataset_found = False
for dataset in dataset_list:
    if dataset.name == dataset_name:
        dataset_id = dataset.id
        dataset_found = True

if not dataset_found:
    print("Dataset with name " + dataset_name + " doesn't exist")

# Download
i = 0
while True:
    print("Getting annotation from page_no: ", i)
    i += 1
    data_list = x1_client.query_data_under_dataset(dataset_id=dataset_id, page_no=i)['datas']
    if not data_list:
        break
    
    data_ids = [x['id'] for x in data_list]
    # annotation
    print("Download annotation")
    annotation_result = x1_client.query_data_and_result(dataset_id=dataset_id, data_ids=data_ids)
    if annotation_result == -1:
        continue
    # save annotation
    annotation_result.to_json(
        export_folder=os.path.join(dataset_dir, 'result')
    )

    # data
    # change data to export format
    for data_id in data_ids:
        data_dict = x1_client.query_data(data_id)[0]
        file_name = data_dict['name']
        export_data_dict = {
            "dataId": data_dict['id'],
            "version": "Xtreme1 v0.6",
            "name": file_name,
            "type": "LIDAR_FUSION"}
        lidar_point_cloud_dict_list = []
        camera_image_dict_list = []
        for content in data_dict['content']:
            if 'camera_config' in content['name']:
                content_info = content['files'][0]['file']
                export_data_dict['cameraConfig'] = {
                    "filename": content_info['name'],
                    "url": content_info['url'],
                    "zipPath": content_info['zipPath']}
            elif 'lidar_point_cloud' in content['name']:
                content_info = content['files'][0]['file']
                lidar_point_cloud_dict = {
                    "filename": content_info['name'],
                    "url": content_info['url'],
                    "zipPath": content_info['zipPath']
                }
                lidar_point_cloud_dict_list.append(lidar_point_cloud_dict)
            elif 'camera_image' in content['name']:
                content_info = content['files'][0]['file']
                camera_image_dict = {
                    "width": content_info['extraInfo']['width'],
                    "height": content_info['extraInfo']['height'],
                    "filename": content_info['name'],
                    "url": content_info['url'],
                    "zipPath": content_info['zipPath']
                }
                camera_image_dict_list.append(camera_image_dict)

        export_data_dict['lidarPointClouds'] = lidar_point_cloud_dict_list
        export_data_dict['cameraImages'] = camera_image_dict_list

        data_path = os.path.join(dataset_dir, 'data')
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
        json_file = os.path.join(data_path, file_name + '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(export_data_dict, f, indent=1, ensure_ascii=False)

    # Image, lidar, config
    for data_id in data_ids:
        print("Download data id: ", data_id)
        x1_client.download_data(
            output_folder=os.path.join(dataset_dir, 'raw_data'), 
            dataset_id=dataset_id,
            data_id=data_id)
