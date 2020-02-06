import os,boto3 

boto3_connection = boto3.resource('s3')

data_path = '../../../../DataScienceProjects/handwrittenmathsymbols/extracted_images/'

all_class_folders = os.listdir(data_path)

for c in all_class_folders:
    dirpath= os.path.join(data_path,c)
    print(dirpath)

s3 = boto3.client('s3')
count = 1
# while count < 5:
#     for folder_name in os.listdir(data_path):
#         if folder_name == '.DS_Store':
#             continue
#         folder_path = os.path.join(data_path,folder_name)
#         for img in os.listdir(folder_path):
#             print('img: ', os.path.abspath(img))
#             with open(os.path.join(folder_path,img),'rb') as f:
#                 print(f)
#                 # s3.upload_fileobj(f, f'capstone2-data-bucket/{folder_name}', os.path.basename(img))
#             count+=1
#             if count > 5:
#                 break
print(data_path)
for dir in os.listdir(data_path):
    if dir != '.DS_Store':
        class_dir = os.path.join(data_path,dir)
        for img in os.listdir(class_dir):
            print(os.path.abspath(img))


