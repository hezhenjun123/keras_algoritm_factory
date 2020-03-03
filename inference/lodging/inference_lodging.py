import logging
import cv2
import os
from inference.inference_base import InferenceBase
import numpy as np
import pdb 
from sklearn.metrics import confusion_matrix, classification_report

logging.getLogger().setLevel(logging.INFO)


#FIXME: I recommend not going through matplotlib interface and doing the numpy concatenations directly. Matplotlib is very slow
#FIXME: Instead we should make a utility function that we can use here and in ImageSummary
class InferenceLodging(InferenceBase):

    def __init__(self, config):
        super().__init__(config)
        self.pred_image_dir = config["INFERENCE"]["PRED_IMAGE_DIR"]
        self.num_process_image = config["INFERENCE"]["NUM_PROCESS_IMAGE"]
        self.evaluate = config["INFERENCE"]["EVALUATE"]
        self.contour_threshold = config["INFERENCE"]["CONTOUR_AREA_THRESHOLD"]
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
        
        colors = {"labels":(0,255,0), "predicted":(0,0,255)}
        def process_contours(original_image, seg_map, image_name="predicted"):
            contours, _ = cv2.findContours(seg_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            newimg = np.copy(original_image)
            res_flag = False
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < self.contour_threshold:
                    continue 
                else:
                    res_flag = True
                print(f"predicted area {idx}", area)
                cv2.drawContours(newimg, contour, -1, colors[image_name], 2)
            if image_name == "predicted":
                predicted.append(1 if res_flag else 0)
            else:
                gt.append(1 if res_flag else 0)
            contour_dir = os.path.join(save_dir_contour,
                                                f"image{count:05d}_{image_name}.png")
            image = cv2.putText(newimg, image_name, (500, 500), cv2.FONT_HERSHEY_SIMPLEX , 1, colors[image_name], 2, cv2.LINE_AA) 
            cv2.imwrite(contour_dir, newimg)

        inference_dataset = dataset
        save_dir_model = os.path.join(self.save_dir, self.pred_image_dir, "model")
        save_dir_contour = os.path.join(self.save_dir, self.pred_image_dir,
                                                 "contour")
        save_dir_original = os.path.join(self.save_dir, self.pred_image_dir, "original")
        save_dir_segmap = os.path.join(self.save_dir, self.pred_image_dir, "predicted-segmap")

        os.makedirs(save_dir_model, exist_ok=True)
        os.makedirs(save_dir_contour, exist_ok=True)
        os.makedirs(save_dir_original, exist_ok=True)
        os.makedirs(save_dir_segmap, exist_ok=True)

        count = 0
        gt = []
        predicted = []
        for elem in inference_dataset:
            pred_res = model.predict(elem[0])
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

            ## predicted 
            process_contours(original_image, resized_pred_seg, "predicted")

            ## labels 
            original_labels = np.squeeze(elem[3], axis=0)
            process_contours(original_image, original_labels, "labels")

            segmap_dir = os.path.join(save_dir_segmap, f"image{count:05d}_segmap.png")
            cv2.imwrite(segmap_dir, resized_pred_seg)

            count += 1
            if count >= self.num_process_image: break
        gt = np.array(gt)
        predicted = np.array(predicted)
        acc = sum(gt == predicted)*1.0/len(gt)
        matrix = confusion_matrix(gt, predicted)
        report = classification_report(gt, predicted)
        wrong_predictions = np.where(predicted != gt)
        print(f"Wrong predictions:{wrong_predictions[0]}")
        print(f"Inferece Accuracy: {acc}")
        print(f"Confusion Matrix:\n {matrix}")
        print(f"Classification Report:\n {report}")

