import pandas as pd
import os
import cv2

root_folder = "yield-level/"
tmp_folder = "yield-level/processed/"
train_csv = os.path.join(root_folder, "train.csv")

train_data = pd.read_csv(train_csv, sep=",").fillna("")
# print(train_data)
new_df_list = []
for index, row in train_data.iterrows():
    img = cv2.imread(os.path.join(root_folder, row["image_path"]))
    image_label = row["image_level_label"]
    print(f"{image_label:02d}")
    new_image_name = f"{image_label:02d}_" + row["image_path"].split('/')[-1]
    print(new_image_name)
    new_image_path = os.path.join(tmp_folder, new_image_name)
    cv2.imwrite(new_image_path, img)



