from xtreme1.client import Client
import argparse
import os
import zipfile

x1_client = Client(
    base_url='http://localhost:8190', 
    access_token='eyJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJCYXNpY0FJIiwiaWF0IjoxNzU0OTY1OTM1LCJzdWIiOiIyIn0.IfNp3s4GdC24hruNIvGymUgqNX0irdtQoWUxfQi3pYPCipmtVtCsL9iyOADhYrrXbUGv1ep09b5LRqq4nkjH1w'
)

parser = argparse.ArgumentParser(description='Process dataset parameters.')

parser.add_argument('--data_path', type=str, required=True,
                    help='Path to the dataset (mandatory)')
parser.add_argument('--dataset_name', type=str, default='test',
                    help='Name of the dataset (default: test)')
parser.add_argument('--task', type=str, default='LIDAR_FUSION',
                    help='Task type (default: LIDAR_FUSION)')

args = parser.parse_args()

data_path = args.data_path
dataset_name = args.dataset_name
task = args.task

# Query a list of datasets
dataset_list, total = x1_client.query_dataset()

dataset_exist = False
dataset_id = -1
for dataset in dataset_list:
    if dataset.name == dataset_name:
        print("Using existing dataset: " + dataset_name)
        dataset_id = dataset.id
        dataset_exist = True

if not dataset_exist:
    print("Making new dataset: " + dataset_name)
    # Create dataset
    dataset = x1_client.create_dataset(
        name=dataset_name, 
        annotation_type=task, 
        description=dataset_name
    )
    dataset_id = dataset.id

zip_file_path = ''
if os.path.isdir(data_path):
    print("data_path is directory. Making zip folder to upload xtreme1")
    zip_path = data_path.rstrip(os.sep)  # remove trailing slash if any
    zip_file_path = zip_path + '.zip'
    base_folder_structure = ['camera_config', 'camera_image_', 'lidar_point_cloud_', 'result']
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for foldername, subfolders, filenames in os.walk(data_path):
            for folder_structure in base_folder_structure:
                if folder_structure in os.path.basename(foldername):
                    archive_folder = os.path.join(
                        os.path.basename(zip_path),
                        os.path.basename(foldername))

                    for filename in filenames:
                        src_file = os.path.join(foldername, filename)
                        arc_file = os.path.join(archive_folder, filename)
                        zf.write(src_file, arc_file)
    data_path = zip_file_path

print("Uploading data from " + data_path + " to dataset " + dataset_name)
response = x1_client.upload_data(
    data_path=data_path,   # could be a single file or a folder
    dataset_id=dataset_id,
    is_local=True
)
print("Upload done")

if zip_file_path != '':
    print("removing zip file: " + zip_file_path)
    os.remove(zip_file_path)
    print("Removed zip file")
