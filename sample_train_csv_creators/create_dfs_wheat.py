import pandas as pd
import os
import numpy as np
import logging
import random

logging.getLogger().setLevel(logging.INFO)

images = []
class_labels = {'low': 0, 'medium': 1, 'high': 2}
# base_path = '/Users/Bo/Desktop/Data/wheat/wheat0605/wheat-vibration-table-data/'
base_path = '/Users/Bo/Desktop/Data/wheat/wheat-vibration-0606/'
train_folder = "chaff-training-set"
valid_folder = "chaff-test-set"

for folder in [train_folder, valid_folder]:
    path = os.path.join(base_path, folder)
    if folder == train_folder:
        split = "train"
    elif folder == valid_folder:
        split = "valid"
    for cam in os.listdir(path):
        if cam.startswith("cam") == False: continue
        path_cams = os.path.join(path, cam)
        for class_name in os.listdir(path_cams):
            path_classes = os.path.join(path_cams, class_name)
            if class_name not in class_labels:
                logging.info("missing labels: {}".format(class_name))
                continue
            class_index = class_labels[class_name]
            for img_folder in os.listdir(path_classes):
                if img_folder.endswith("bag") == False:
                    logging.info("missing image folder: {}".format(img_folder))
                    continue
                path_img_folder = os.path.join(path_classes, img_folder)
                for img in os.listdir(path_img_folder):
                    if img.endswith('.png'):
                        path_img = (path_img_folder + '/' + img).replace(base_path, "")
                        images.append([[class_name], path_img, [class_index], split])

random.shuffle(images)
df = pd.DataFrame(images, columns=['label_names', 'image_path', 'labels', 'split'])
print(f'Total images detected {len(df)}')
df.to_parquet("train_vibration0606.pt")
