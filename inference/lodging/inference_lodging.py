import logging
import cv2
import os
from inference.inference_base import InferenceBase
import numpy as np

logging.getLogger().setLevel(logging.INFO)


#FIXME: I recommend not going through matplotlib interface and doing the numpy concatenations directly. Matplotlib is very slow
#FIXME: Instead we should make a utility function that we can use here and in ImageSummary
class InferenceLodging(InferenceBase):

    def __init__(self, config):
        super().__init__(config)
        self.pred_image_dir = config["INFERENCE"]["PRED_IMAGE_DIR"]
        self.num_process_image = config["INFERENCE"]["NUM_PROCESS_IMAGE"]
        self.evaluate = config["INFERENCE"]["EVALUATE"]
        if self.num_process_image >= 99999:
            raise Exception("cannot process more than 99999 images ")

    def run_inference(self):
        inference_transform = self.generate_transform()
        inference_data_split = self.read_train_csv()
        inference_dataset = self.generate_dataset(inference_data_split, inference_transform, self.evaluate)
        model = self.load_model()
        if self.evaluate:
            self.__produce_evaluation(model, inference_dataset)
        else:
            self.__produce_segmentation_image(model, inference_dataset)
        logging.info("================Inference Complete=============")

    def __produce_evaluation(self, model, dataset):
        logging.info("================Evaluation Starts=============")
        model.evaluate(dataset)
        logging.info("================Evaluation Completes=============")
        
    def __produce_segmentation_image(self, model, dataset):
        inference_dataset = dataset
        save_dir_model = os.path.join(self.save_dir, self.pred_image_dir, "model")
        save_dir_original_contour = os.path.join(self.save_dir, self.pred_image_dir,
                                                 "original-contour")
        save_dir_original = os.path.join(self.save_dir, self.pred_image_dir, "original")
        save_dir_segmap = os.path.join(self.save_dir, self.pred_image_dir, "original-segmap")

        os.makedirs(save_dir_model, exist_ok=True)
        os.makedirs(save_dir_original_contour, exist_ok=True)
        os.makedirs(save_dir_original, exist_ok=True)
        os.makedirs(save_dir_segmap, exist_ok=True)

        count = 0
        for elem in inference_dataset:
            pred_res = model.predict(elem)
            transformed_image = np.squeeze(elem[0], axis=0)
            original_image = np.squeeze(elem[1], axis=0)
            pred_p = np.squeeze(pred_res, axis=(0, 3))
            pred_seg = pred_p
            threshold = 0.5
            pred_seg[pred_seg > threshold] = 1
            pred_seg[pred_seg <= threshold] = 0

            resize_shape = (original_image.shape[1], original_image.shape[0])
            resized_pred_seg = cv2.resize(np.float32(pred_seg),
                                          resize_shape,
                                          interpolation=cv2.INTER_NEAREST).astype(np.uint8)

            logging.info(f"processed image: {count:05d}")

            original_image = original_image[:, :, ::-1]
            original_dir = os.path.join(save_dir_original, f"image{count:05d}_original.png")
            cv2.imwrite(original_dir, original_image)

            contours, _ = cv2.findContours(resized_pred_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            newimg = np.copy(original_image)
            for contour in contours:
                cv2.drawContours(newimg, contour, -1, (0, 0, 255), 2)
            original_contour_dir = os.path.join(save_dir_original_contour,
                                                f"image{count:05d}_original_contour.png")
            cv2.imwrite(original_contour_dir, newimg)

            segmap_dir = os.path.join(save_dir_segmap, f"image{count:05d}_segmap.png")
            cv2.imwrite(segmap_dir, resized_pred_seg)

            count += 1
            if count >= self.num_process_image: break
