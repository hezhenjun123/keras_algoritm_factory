import subprocess
import tensorflow as tf
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)
import cv2
import albumentations as A
import glob

# model_path = "run_025/checkpoints/epoch_0240/cp.ckpt"
# model_path = "model_vibration/run_006/checkpoints/epoch_0050/cp.ckpt"
# model_path = "model_vibration/run_007/checkpoints/epoch_0100/cp.ckpt"
model_path = "model_vibration/run_008/checkpoints/epoch_0010/cp.ckpt"
# image_path = "chaff-test-set/cam1/low/chaff1_camera_843112070583_1558132150.6690688.bag_Color_103.png"
image_files = [
    "chaff-test-set/cam1/low/*.png", "chaff-test-set/cam1/medium/*.png",
    "chaff-test-set/cam1/high/*.png"
]

model = tf.keras.models.load_model(model_path)
model.summary()
transform = A.Compose([A.RandomCrop(388, 388), A.Resize(300, 300)])
# transform = A.Compose([A.Resize(300,300)])
# transform = A.Compose([A.Resize(400,600)])
num_total = 0
num_incorrect = 0
label = 0
for image_file in image_files:
    logging.info("==============================================")
    label += 1
    for file in glob.glob(image_file):
        image = cv2.imread(file)
        img_transform = transform(image=image)["image"]
        img_transform = np.expand_dims(img_transform, axis=0)
        pred_res = model.predict(x=img_transform)
        # logging.info("pred results: {}".format(pred_res))
        logging.info("pred label: {}, file: {}".format(np.argmax(pred_res) + 1, file))
        num_total += 1
        if np.argmax(pred_res) + 1 != label: num_incorrect += 1
logging.info("total num: {}, incorrect: {}".format(num_total, num_incorrect))
