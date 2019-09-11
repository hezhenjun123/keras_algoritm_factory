import logging
import cv2
import os
from inference.inference_base import InferenceBase
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger().setLevel(logging.INFO)


#FIXME: I recommend not going through matplotlib interface and doing the numpy concatenations directly. Matplotlib is very slow
#FIXME: Instead we should make a utility function that we can use here and in ImageSummary
class InferenceChaff(InferenceBase):

    def __init__(self, config):
        super().__init__(config)
        self.pred_image_dir = config["INFERENCE"]["PRED_IMAGE_DIR"]
        self.num_process_image = config["INFERENCE"]["NUM_PROCESS_IMAGE"]
        if self.num_process_image >= 99999:
            raise Exception("cannot process more than 99999 images ")

    def run_inference(self):
        inference_transform = self.generate_transform()
        inference_data_split = self.read_train_csv()
        inference_dataset = self.generate_dataset(inference_data_split, inference_transform)
        model = self.load_model()
        self.__produce_segmentation_image(model, inference_dataset)
        logging.info("================Inference Complete=============")

    def __produce_segmentation_image(self, model, dataset):
        inference_dataset = dataset.unbatch().batch(1)
        save_dir = os.path.join(self.save_dir, self.pred_image_dir)
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        count = 0
        for elem in inference_dataset:
            pred_res = model.predict(elem)
            original_image = np.squeeze(elem[2], axis=0)
            resize_shape = (original_image.shape[1], original_image.shape[0])

            transformed_seg = np.squeeze(elem[1], axis=0)
            original_seg = cv2.resize(np.float32(transformed_seg),
                                      resize_shape,
                                      interpolation=cv2.INTER_NEAREST)

            pred_seg = np.round(np.squeeze(pred_res, axis=(0, 3)))
            resized_pred_seg = cv2.resize(np.float32(pred_seg),
                                          resize_shape,
                                          interpolation=cv2.INTER_NEAREST)

            fig1 = plt.figure()
            fig1.add_subplot(2, 2, 1)
            plt.imshow(original_image)

            fig1.add_subplot(2, 2, 2)
            plt.imshow(original_seg, cmap='gray', vmin=0, vmax=1)

            fig1.add_subplot(2, 2, 3)
            plt.imshow(pred_seg, cmap='gray')

            fig1.add_subplot(2, 2, 4)
            plt.imshow(original_image)
            plt.contour(resized_pred_seg)
            plt.savefig(os.path.join(save_dir, f"image{count:05d}_model.png"))

            fig2 = plt.figure()
            plt.imshow(original_image)
            plt.contour(resized_pred_seg)
            plt.savefig(os.path.join(save_dir, f"image{count:05d}_original.png"))

            count += 1
            if count >= self.num_process_image: break
