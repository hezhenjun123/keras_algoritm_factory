import pandas as pd
import os
import numpy as np
import logging
import random

logging.getLogger().setLevel(logging.INFO)

images = []
class_labels = {'low': 0, 'high': 1}
base_path = '/Users/Bo/Desktop/GoogleDrive/workspace/bo-utility/output_saved/yield-level/'
# base_path = '/Users/Bo/Desktop/Data/wheat/wheat-vibration-0606/'
train_folder = "chaff-training-set"
valid_folder = "chaff-test-set"

for folder in [train_folder, valid_folder]:
    path = os.path.join(base_path, folder)
    if folder == train_folder:
        split = "train"
    elif folder == valid_folder:
        split = "valid"

    for class_name in os.listdir(path):
        path_classes = os.path.join(path, class_name)
        if class_name not in class_labels:
            logging.info("missing labels: {}".format(class_name))
            continue
        class_index = class_labels[class_name]
        for img in os.listdir(path_classes):
            if img.endswith('.png'):
                path_img = (path_classes + '/' + img).replace(base_path, "")
                images.append([[class_name], path_img, [class_index], split])

random.shuffle(images)
df = pd.DataFrame(images, columns=['label_names', 'image_path', 'labels', 'split'])
print(f'Total images detected {len(df)}')
df.to_parquet("data/train_yield_level_0624.pt")