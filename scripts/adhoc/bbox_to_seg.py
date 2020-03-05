import os
import xmltodict
import cv2
import pandas as pd
import numpy as np


def parse_xml(label_path):
    with open(label_path, "rb") as f:
        lbl = xmltodict.parse(f.read())

        # Read bounding box
    bboxes = []
    if "object" in lbl["annotation"].keys():  # If there is (are) bbox(es)
        object_annotation = lbl["annotation"]["object"]
        if not type(object_annotation) == list:
            object_annotation = [object_annotation]
        for bbox_dict in object_annotation:
            bndbox_dict = bbox_dict["bndbox"]
            xmin = int(float(bndbox_dict["xmin"]))
            ymin = int(float(bndbox_dict["ymin"]))
            xmax = int(float(bndbox_dict["xmax"]))
            ymax = int(float(bndbox_dict["ymax"]))
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin, bbox_dict["name"]]
            bboxes.append(bbox)

    return bboxes


root_folder = "breakage-data-v0.1/"
tmp_folder = "breakage-data-v0.1/tmp/"
train_csv = os.path.join(root_folder, "train.csv")

train_data = pd.read_csv(train_csv, sep=",").fillna("")
# print(train_data)
new_df_list = []
for index, row in train_data.iterrows():
    img = cv2.imread(os.path.join(root_folder, row["image_path"]))
    bbox = parse_xml(os.path.join(root_folder, row["bbox_label_path"]))
    seg_img = np.zeros((img.shape[0], img.shape[1], 1))
    for xmin, ymin, xlength, ylength, _ in bbox:
        seg_img[ymin:ymin + ylength, xmin:xmin + xlength, 0] = 1
        img[ymin:ymin + ylength, xmin:xmin + xlength, 1] = 128
    cv2.imwrite(os.path.join(tmp_folder, row["bbox_label_path"].split("/")[-1]) + ".png", img)
    seg_path = os.path.join(root_folder, row["bbox_label_path"]) + ".png"
    row["seg_label_path"] = row["bbox_label_path"] + ".png"
    cv2.imwrite(seg_path, seg_img)
    new_df_list.append(row)
    # print(row["seg_label_path"])

new_df = pd.DataFrame(new_df_list)
new_train_path = os.path.join(root_folder, "seg_train.csv")
new_df.to_csv(new_train_path)
# for index, row in new_df.iterrows():
#     print(row)
# print(train_data)
