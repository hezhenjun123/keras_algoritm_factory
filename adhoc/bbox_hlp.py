import os
import json
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

path1 = "/Users/Bo/Desktop/GoogleDrive/workspace/breakage-hlp/1a0c6a37-e778-4c29-8eb4-30ae6043f891/img"
path2 = "/Users/Bo/Desktop/GoogleDrive/workspace/breakage-hlp/e9e41011-f4ea-4a19-896c-3453194b4365/img"


def compute_iou(y_true, y_pred):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(np.float32)
    # compute mean iou
    intersection = np.diag(current).copy()
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    #prevent division by 0
    intersection[union == 0] = 1e-10
    union[union == 0] = 1e-10
    IoU = intersection / union
    return IoU.mean()


def get_pred(json_path, image_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    img = cv2.imread(image_path)
    pred = np.zeros((img.shape[0], img.shape[1]))
    # print(img.shape)
    for key, val in json_data["labels"].items():
        start_x = round(val["DATA"]["starting_point_x"])
        start_y = round(val["DATA"]["starting_point_y"])
        end_x = round(val["DATA"]["ending_point_x"])
        end_y = round(val["DATA"]["ending_point_y"])
        # print(start_x, start_y, end_x, end_y)
        pred[start_y:end_y, start_x:end_x] = 1
    return pred


count = 0
total_iou = 0
for i in range(1, 11):
    file1 = os.path.join(path1, f"{i}.jpg.meta.json")
    file2 = os.path.join(path2, f"{i}.jpg.meta.json")
    image_path = os.path.join(path1, f"{i}.jpg")
    print(i)
    pred1 = get_pred(file1, image_path)
    pred2 = get_pred(file2, image_path)
    mean_iou = compute_iou(pred1, pred2)
    count += 1
    total_iou += mean_iou
    print(mean_iou)
    print('\n')

total_mean_iou = total_iou / count
print(f"hlp iou: {total_mean_iou}")

# print(json_data1["labels"])
# print(img.shape)
# print(pred1.shape)
# for key, val in json_data1["labels"].items():
#     print(val["DATA"])
