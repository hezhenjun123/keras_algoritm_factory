import logging
import cv2
import os
import pdb 
from inference.inference_base import InferenceBase
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utilities.file_system_manipulation import directory_to_file_list
import tensorflow as tf
tf.enable_eager_execution()


class InferenceYieldAbsoluteNewView(InferenceBase):

    def __init__(self, config):
        super().__init__(config)
        self.evaluate = config[config["INFERENCE_ENGINE"]]["EVALUATE"]
        self.inference_visualize = config[config["INFERENCE_ENGINE"]]["VISUALIZE"]
        
    def run_inference(self):
        inference_transform = self.generate_transform()
        inference_data_split = self.read_train_csv()
        inference_dataset = self.generate_dataset(inference_data_split, inference_transform, self.evaluate)
        model = self.load_model()

        if self.evaluate:
            self.__produce_evaluation(model, inference_dataset)
        else:
            pass
        logging.info("================Inference Complete=============")
    
    def __produce_evaluation(self, model, dataset):
        logging.info("================Evaluation Starts=============")
        model.evaluate(dataset)
        logging.info("================Evaluation Completes=============")