import platform
import os
import logging
from transforms.transform_factory import TransformFactory
from model.model_factory import ModelFactory
import tensorflow as tf
from data_generators.generator_factory import DataGeneratorFactory
import shutil
from tensorflow.python.tools import freeze_graph
import numpy as np
import cv2
if platform.machine() != 'aarch64':
    import pandas as pd
    from sklearn.metrics import confusion_matrix, classification_report

logging.getLogger().setLevel(logging.INFO)


#FIXME: May want to combine Inference Base with Experiment base. Experiment can handle both training and inference
class InferenceBase:

    def __init__(self, config):
        self.config = config
        self.split_col = config[config["INFERENCE_ENGINE"]]["SPLIT"]
        self.split_val = config[config["INFERENCE_ENGINE"]]["SPLIT_VAL"]
        self.inference_csv = config[config["INFERENCE_ENGINE"]]["INFERENCE_CSV_FILE"]
        self.csv_separator = config[config["INFERENCE_ENGINE"]]["SEPARATOR"]
        self.inference_transform_name = self.config[config["INFERENCE_ENGINE"]]["TRANSFORM"]
        self.inference_generator_name = self.config[config["INFERENCE_ENGINE"]]["GENERATOR"]
        self.model_name = self.config[config["INFERENCE_ENGINE"]]["MODEL_NAME"]
        self.num_classes = config["NUM_CLASSES"]
        self.save_dir = config["DIR_OUT"]
        self.load_model_path = self.config["LOAD_MODEL_DIRECTORY"]
        if config["RUN_ENV"] == "local":
            self.__local_override_config(config)

    def __local_override_config(self, config):
        self.save_dir = self.__create_run_dir(config["LOCAL_OVERRIDE"]["DIR_OUT"])

    def generate_transform(self):
        transform_factory = TransformFactory(self.config)
        inference_transform = transform_factory.create_transform(self.inference_transform_name)
        return inference_transform

    def read_train_csv(self):
        data_from_inference_csv = pd.read_csv(self.inference_csv, sep=self.csv_separator).fillna("")
        logging.info(data_from_inference_csv.head())
        logging.info("#" * 15 + "Reading inference data" + "#" * 15)
        if self.split_val not in ["all", "train", "valid"]:
            raise ValueError("spilt_value={} is not allowed, only ['train', 'valid', 'all'] are supported.".format(self.split_val))
        if self.split_val == "all":
            return data_from_inference_csv
        inference_data_filter = data_from_inference_csv[self.split_col] == self.split_val
        data_inference_split = data_from_inference_csv[inference_data_filter]
        return data_inference_split

    def generate_dataset(self, data_inference_split, inference_transform, evaluate):
        self.config['BATCH_SIZE'] = 1
        generator_factory = DataGeneratorFactory(self.config)
        inference_generator = generator_factory.create_generator(self.inference_generator_name)
        if evaluate:
            inference_dataset = inference_generator.create_dataset(
            df=data_inference_split, transforms=inference_transform)
        else:
            inference_dataset = inference_generator.create_inference_dataset(
            df=data_inference_split, transforms=inference_transform)
        return inference_dataset

    def load_model(self, create_raw_model = False):
        if not self.config["LOAD_MODEL"]:
            raise ValueError('LOAD_MODEL config must be set to true for inference')
        if create_raw_model:
            self.config["LOAD_MODEL"] = False
        model_factory = ModelFactory(self.config)
        model = model_factory.create_model(self.model_name)
        return model

    def freeze_to_pb(self, save_dir):
        tf.compat.v1.disable_eager_execution()
        shutil.rmtree(save_dir) if os.path.exists(save_dir) else None
        os.makedirs(save_dir)
        model = self.load_model()
        model.model.save_weights(save_dir + "/tmp_model_weights.h5")
        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(0)
        model = self.load_model(False)
        model.model.load_weights(save_dir + "/tmp_model_weights.h5")
        shutil.rmtree(save_dir) if os.path.exists(save_dir) else None
        tf.saved_model.simple_save(tf.keras.backend.get_session(),
                                   save_dir,
                                   inputs={"input": model.model.inputs[0]},
                                   outputs={"output": model.model.outputs[0]})

        freeze_graph.freeze_graph(None,
                                  None,
                                  None,
                                  None,
                                  model.model.outputs[0].op.name,
                                  None,
                                  None,
                                  os.path.join(save_dir + "/frozen_model.pb"),
                                  False,
                                  "",
                                  input_saved_model_dir=save_dir)
    def run_inference(self):
        inference_transform = self.generate_transform()
        inference_data_split = self.read_train_csv()
        inference_dataset = self.generate_dataset(inference_data_split, inference_transform, self.evaluate)
        model = self.load_model()
        if self.evaluate:
            self._produce_evaluation(model, inference_dataset)
        else:
            self._inference(model, inference_dataset)
        logging.info("================Inference Complete=============")

    def _produce_evaluation(self, model, dataset):
        logging.info("================Evaluation Starts=============")
        model.evaluate(dataset)
        logging.info("================Evaluation Completes=============")
    
    def _inference(self, model, dataset):
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


    def __create_run_dir(self, save_dir):
        """Creates a numbered directory named "run1". If directory "run1" already
        exists then creates directory "run2", and so on.

        Parameters
        ----------
        save_dir : str
            The root directory where to create the "run{number}" folder.

        Returns
        -------
        str
            The full path of the newly created "run{number}" folder.
        """
        tf.io.gfile.mkdir(save_dir)
        list_of_files = tf.io.gfile.listdir(save_dir)
        i = 1
        while "inference{}".format(i) in list_of_files:
            i += 1
        run_dir = os.path.join(save_dir, "inference{}".format(i))
        tf.io.gfile.mkdir(run_dir)
        logging.info("#" * 40)
        logging.info("Saving inference on {}".format(run_dir))
        logging.info("#" * 40)
        return run_dir
