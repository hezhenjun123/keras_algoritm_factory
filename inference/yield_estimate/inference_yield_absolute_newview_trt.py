import logging
import cv2
import os
from inference.inference_base import InferenceBase
import numpy as np
import tensorflow as tf
from inference.common import allocate_buffers, do_inference, TRT_LOGGER, ModelData, build_engine
from inference.common import ModelInferTF
from utilities.file_system_manipulation import directory_to_file_list
from utilities.helper import stream_video
from utilities.s3context import CACHE

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import uff

class InferenceYieldAbsoluteNewViewTRT(InferenceBase):

    def __init__(self, config):
        super().__init__(config)
        self.trt_engine_path = config["INFERENCE"]["TRT_ENGINE_PATH"]
        self.video_path = config["INFERENCE"]["VIDEO_PATH"]
        self.inference_transform = self.generate_transform()
        self.create_engine = config["INFERENCE"]["CREATE_ENGINE"]
        self.input_size = config["TRANSFORM"]["RESIZE"]
        self.input_channel = config["DATA_GENERATOR"]["OUTPUT_IMAGE_CHANNELS"]
        self.pb_file_path = config["INFERENCE"]["PB_FILE_PATH"]
        self.input_name = config["INFERENCE"]["INPUT_NAME"]
        self.output_name = config["INFERENCE"]["OUTPUT_NAME"]
        self.num_frames = config["INFERENCE"]["NUM_FRAMES"]
        if self.create_engine == True:
            model_data = ModelData(self.pb_file_path,
                                   self.input_name,
                                   (self.input_channel, self.input_size[0], self.input_size[1]),
                                   self.output_name,
                                   config["INFERENCE"]["FP16_MODE"],
                                   self.trt_engine_path
            )
            self.build_and_dump_engine(model_data)

        self.batch_size = 1
        try:
            self.engine = self.load_engine(self.trt_engine_path)
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        except:
            print("Failed to load engine")

    #TODO(yuanzhedong): move to base, may need trt deps?
    def build_and_dump_engine(self, model_data):
        with build_engine(model_data) as engine:
            print('Engine:', engine)
            buf = engine.serialize()
            with open(model_data.trt_engine_path, 'wb') as f:
                f.write(buf)

    def load_engine(self, engine_file):
        with open(engine_file, "rb") as f:
            engine_data = f.read()
            engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_data)
            return engine

    def run_inference(self):
        file_list = sorted(directory_to_file_list(self.video_path))
        file_count = 1
        total_count = len(file_list)
        output = []
        for file_path in file_list:
            for idx, frame in enumerate(stream_video(CACHE.fetch(file_path))):
                output.append(self.get_image_pred(frame, True))
                if idx > self.num_frames:
                    break
        logging.info("================Inference Complete=============")
        return output

    def run_inference_tf(self):
        file_list = sorted(directory_to_file_list(self.video_path))
        file_count = 1
        total_count = len(file_list)
        model = ModelInferTF(self.pb_file_path, self.input_name, self.output_name)
        output = []
        for file_path in file_list:
            for idx, frame in enumerate(stream_video(CACHE.fetch(file_path))):
                frame, _ = self.inference_transform.apply_transforms(frame, frame[:,:,1])
                frame = np.expand_dims(frame, axis=0)
                output.append(model.infer(frame))
                print(output[-1])
                if idx > self.num_frames:
                    break
        return output

    def get_image_pred(self, image, is_raw = False):

        if is_raw:
            #TODO: Don't need to transform labels here
            image, _ = self.inference_transform.apply_transforms(image, image[:,:,1])


        if type(image).__module__ == np.__name__:
            image = image.transpose((2, 0, 1)).ravel()
        else:
            raise ValueError("Could not handle input type: {}".format(type(image)))

        np.copyto(self.inputs[0].host, image)
        [output] = do_inference(context = self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        print(output)
        #TODO(yuanzhedong): check consistancy
        return np.copy(output)
