import tensorflow as tf
import logging
from data_generators.generator_base import DataGeneratorBase
import numpy as np
from utilities import file_system_manipulation as fsm
import cv2
import os
logging.getLogger().setLevel(logging.INFO)


class GeneratorVideo(DataGeneratorBase):

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = config["BATCH_SIZE"]
        self.num_parallel_calls = 1

    def __create_video_buffer(self, video_path):
        if fsm.is_s3_path(video_path):
            video_path = fsm.s3_to_local(video_path, './video.avi')[0]
        if not os.path.exists(video_path):
            raise ValueError("Incorrect video path")
        self.video_buffer = cv2.VideoCapture(video_path)

    def create_inference_dataset(self, df, transforms=None):
        video_path = df
        self.__create_video_buffer(video_path)
        n_frames = int(self.video_buffer.get(cv2.CAP_PROP_FRAME_COUNT))
        dataset = tf.data.Dataset.from_tensor_slices(dict(img=np.arange(n_frames)))
        dataset = dataset.map(self.__load_data(), num_parallel_calls=1)
        if transforms != None and transforms.has_transform():
            transforms_map = self.__get_transform_map(transforms, self.output_shape,
                                                      self.output_image_channels,
                                                      self.output_image_type)
            dataset = dataset.map(transforms_map)
            logging.info(dataset)
        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        dataset = dataset.map(lambda row: (row["image"], row["original_image"]))
        dataset = dataset.prefetch(4)
        logging.info("====================inference dataset=====================")
        logging.info(dataset)
        return dataset

    def __get_transform_map(self, transforms, output_shape, output_image_channels,
                            output_image_type):

        def transform_map(row):
            original_image = row["image"]
            augmented = tf.compat.v1.py_func(transforms.apply_transforms,
                                             [row["image"], row["label"]],
                                             [output_image_type, tf.float64])
            logging.info(augmented)
            image = augmented[0]
            image.set_shape(output_shape + (output_image_channels,))
            logging.info(image)
            row["image"] = image
            row["original_image"] = original_image
            return row

        return transform_map

    def create_dataset(self, df, transforms=None):
        raise NotImplementedError('This Generator is not suitable for training')

    def __load_data(self):

        def load_data(row):
            image = tf.compat.v1.py_func(lambda: self.video_buffer.read()[1], [], [np.uint8])[0]
            image = image[:, :, ::-1]
            new_row = dict(
                image=image,
                label=tf.zeros_like(image,dtype=tf.float64),
            )
            return new_row

        return load_data
