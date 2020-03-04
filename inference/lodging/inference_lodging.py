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
        self.inference_visualize = config["INFERENCE"]["VISUALIZE"]
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
            self.__inference(model, inference_dataset)
        logging.info("================Inference Complete=============")

    def __produce_evaluation(self, model, dataset):
        logging.info("================Evaluation Starts=============")
        model.evaluate(dataset)
        logging.info("================Evaluation Completes=============")
        
    def __inference(self, model, dataset):
        """
        Given model and inference data set in format of (resize_img, original_img, resize_seg_label, original_seg_label), 
        make binary classification and return the predictions. 
        """
        
        def process_contours(original_image, seg_maps):
            """
            Given the original image, and the pair of ground_truth_seg and prediction_seg_map, 
            generate contours and make binary classification.

            Arguments:
                original_image {np.array} -- the original image.
                seg_maps {list} -- [ground_truth_seg_map, predicted_seg_map]

            Returns:
                [list] -- [ground truth, prediction]
            """
            colors = [(0,255,0), (0,0,255)]
            seg_names = ["Ground_truth", "Prediction"]
            res = []
            img_to_draw = []
            for idx, seg_map in enumerate(seg_maps):
                newimg = np.copy(original_image)
                contours, _ = cv2.findContours(seg_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                res_flag = False
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if area < self.contour_threshold:
                        continue 
                    else:
                        res_flag = True
                    logging.info(f"{seg_names[idx]} area {i}: {area}")

                    if self.inference_visualize:
                        cv2.drawContours(newimg, contour, -1, colors[idx], 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        loc = (500, 500)
                        newimg = cv2.putText(newimg, seg_names[idx], loc, font, 1, colors[idx], 2, cv2.LINE_AA) 
                
                res.append(1 if res_flag else 0) # 1 NG; 0 OK.
                img_to_draw.append(newimg)
            
            if self.inference_visualize:
                img_name = f"image{count:05d}_groundtruth_and_prediction.png"
                contour_dir = os.path.join(save_dir_contour,img_name)
                cv2.imwrite(contour_dir, np.concatenate(img_to_draw, axis=1))
            return res

        if self.inference_visualize:
            save_dir_contour = os.path.join(self.save_dir, self.pred_image_dir,
                                                    "contour")
            os.makedirs(save_dir_contour, exist_ok=True)

        count = 0
        gt, predicted = [], []
        inference_dataset = dataset
        for elem in inference_dataset:
            assert(len(elem)==4, "the inference data format is wrong")
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
            original_labels = np.squeeze(elem[3], axis=0)
            res = process_contours(original_image, [original_labels, resized_pred_seg])
            gt.append(res[0])
            predicted.append(res[1])
            
            count += 1
            if count >= self.num_process_image: break

        gt = np.array(gt)
        predicted = np.array(predicted)
        acc = sum(gt == predicted)*1.0/len(gt)
        matrix = confusion_matrix(gt, predicted)
        report = classification_report(gt, predicted)
        wrong_predictions = np.where(predicted != gt)
        logging.info(f"Wrong prediction image indexes:{wrong_predictions[0]}")
        logging.info(f"Inferece accuracy: {acc}")
        logging.info(f"Confusion matrix:\n {matrix}")
        logging.info(f"Classification report:\n {report}")
        return predicted

