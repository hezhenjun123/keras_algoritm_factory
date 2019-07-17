import os
import logging
import random
from boto3 import client

logging.getLogger().setLevel(logging.INFO)

class_labels = {'low': 0, 'medium': 1, 'high': 2}
s3_bucket = "zoomlion-sample"
base_path = "archive/wheat-vibration-0606"
train_valid_split = {"train": "chaff-training-set", "valid": "chaff-test-set"}
conn = client('s3')
ext = ".png"
csv_path = "data/train_vibration0606.csv"

if os.path.exists(csv_path) is True: raise Exception("training csv file exists")
images = []
schema = "image_path\tsegmentation_path\tlabel_name\tlabel\tsplit"
for split, split_folder in train_valid_split.items():
    for label_name, label in class_labels.items():
        s3_prefix = f"{base_path}/{split_folder}/{label_name}"
        object_list = conn.list_objects(Bucket=s3_bucket, Prefix=s3_prefix)['Contents']
        for item in object_list:
            full_path = item["Key"]
            if full_path.endswith(ext) is False: continue
            path_img = full_path.replace(base_path + "/", "")
            images.append(f"{path_img}\t\t{label_name}\t{label}\t{split}")
random.shuffle(images)
with open(csv_path, 'w') as fw:
    fw.write(schema + '\n')
    for line in images:
        fw.write(line + '\n')
