import logging
import argparse
import cv2
import tensorflow as tf
import numpy as np
from utilities.config import read_config
from transforms.transform_segmentation import TransformSegmentation
from transforms.transform_base import TransformBase
from typing import List,Tuple
from tensorflow import DType

logging.getLogger().setLevel(logging.DEBUG)


def __load_data(video_buffer):
    image = tf.compat.v1.py_func(lambda: video_buffer.read()[1], [], [np.uint8])[0]
    image = image[:, :, ::-1]
    new_row = dict(
        image=image,
        label=tf.zeros_like(image, dtype=tf.float64),
    )
    return new_row


def __get_transform_map(transforms, output_shape, output_image_channels,
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


def create_dataset(input_video_path: str, transforms: TransformBase, batch_size: int, output_shape: Tuple[int, int], output_image_channels: int, output_image_type: DType, drop_remainder: bool):
    video_buffer = cv2.VideoCapture(input_video_path)
    n_frames = int(video_buffer.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frame count: ", n_frames)
    dataset = tf.data.Dataset.from_tensor_slices(dict(img=np.arange(n_frames)))
    dataset = dataset.map(lambda x: __load_data(video_buffer), num_parallel_calls=1)
    if transforms.has_transform():
        transforms_map = __get_transform_map(transforms, output_shape, output_image_channels, output_image_type)
        dataset = dataset.map(transforms_map)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.map(lambda row: (row["image"], row["original_image"]))
    dataset = dataset.prefetch(4)
    return dataset


def main():
    config = read_config("model_config_segmentation_sprayerweed.yaml")

    batch_size: int = config["BATCH_SIZE"]
    input_video_path: str = config["INFERENCE"]["VIDEO_PATH"]
    output_image_channels: int = config["DATA_GENERATOR"]["OUTPUT_IMAGE_CHANNELS"]
    output_shape = config["DATA_GENERATOR"]["OUTPUT_SHAPE"]
    output_shape: Tuple[int,int] = (output_shape[0], output_shape[1])
    output_image_type: DType = tf.dtypes.as_dtype(np.dtype(config["DATA_GENERATOR"]["OUTPUT_IMAGE_TYPE"]))
    drop_remainder: bool = config["DATA_GENERATOR"]["DROP_REMAINDER"]

    transforms: TransformSegmentation = TransformSegmentation(config)
    dataset = create_dataset(input_video_path, transforms, batch_size, output_shape, output_image_channels, output_image_type, drop_remainder)
    dataset = dataset.apply(tf.data.experimental.unbatch())
    ei = 0
    for elem in dataset:
        ei += 1
        print("Video element: ", ei)


if __name__ == "__main__":
    main()
