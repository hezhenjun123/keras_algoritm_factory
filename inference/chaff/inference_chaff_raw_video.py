import logging
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from inference.inference_base import InferenceBase
from utilities.file_system_manipulation import directory_to_file_list
import tensorflow as tf
tf.enable_eager_execution()


logging.getLogger().setLevel(logging.INFO)


#FIXME: I recommend not going through matplotlib interface and doing the numpy concatenations directly. Matplotlib is very slow
#FIXME: Instead we should make a utility function that we can use here and in ImageSummary
class InferenceChaffRawVideo(InferenceBase):
    def __init__(self, config):
        super().__init__(config)
        self.pred_image_dir = config["INFERENCE"]["PRED_IMAGE_DIR"]
        self.video_path = config["INFERENCE"]["VIDEO_PATH"]
        self.output_fps = config["INFERENCE"]["OUTPUT_FPS"]

    def run_inference(self):
        inference_transform = self.generate_transform()
        model = self.load_model()

        file_list = sorted(directory_to_file_list(self.video_path))
        total_count = len(file_list)
        for file_count, file_path in enumerate(file_list, start=1):
            logging.info(f"process video num: {file_count}/{total_count}")
            logging.info(f"file name: {file_path}")
            inference_dataset = self.generate_dataset(file_path, inference_transform)
            input_video_name = file_path.split('/')[-1]
            self.output_video_name = f"{input_video_name}.avi"
            self.__produce_segmentation_image(model, inference_dataset)
        logging.info("================Inference Complete=============")

    def __produce_segmentation_image(self, model, dataset):
        #tf 1.14 doesn't have unbatch
        inference_dataset = dataset.batch(4)
        writer = None
        for elem in inference_dataset:
            #no unbatch means we need to drop bad dimension (4,1,512,512,3) -> (4,512,512,3)
            pred_res = model.predict(elem[0][:,0])
            for pred in pred_res:
                out = np.round(pred)[:,:,0].astype(np.uint8)      
                # out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
                # print(out.shape)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_shape = (out.shape[1], out.shape[0])
                    writer = cv2.VideoWriter(os.path.join(self.save_dir, self.output_video_name), fourcc,
                                                self.output_fps, video_shape, False)
                writer.write(out)

        writer.release()
