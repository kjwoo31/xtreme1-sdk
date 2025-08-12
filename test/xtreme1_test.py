from xtreme1.client import Client

x1_client = Client(
    base_url='http://localhost:8190', 
    access_token='eyJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJCYXNpY0FJIiwiaWF0IjoxNzU0OTY1OTM1LCJzdWIiOiIyIn0.IfNp3s4GdC24hruNIvGymUgqNX0irdtQoWUxfQi3pYPCipmtVtCsL9iyOADhYrrXbUGv1ep09b5LRqq4nkjH1w'
)

# Query a list of datasets
dataset_list, total = x1_client.query_dataset()
print("Dataset list: ", dataset_list)
test_dataset = dataset_list[0]
print("Downloading dataset of ", test_dataset)

# Download dataset
x1_client.download_data(
    output_folder='test_dataset', 
    dataset_id=test_dataset.id
)

# Download annotation
i = 0
while True:
    print("Getting annotation from page_no: ", i)
    i += 1
    data_list = x1_client.query_data_under_dataset(dataset_id=test_dataset.id, page_no=i)['datas']
    if not data_list:
        break
    
    data_ids = [x['id'] for x in data_list]
    annotation_result = x1_client.query_data_and_result(dataset_id=test_dataset.id, data_ids=data_ids)
    if annotation_result == -1:
        continue
    # Any further actions
    annotation_result.to_json(
        export_folder='annotation_result'
    )
