import tensorflow as tf
import numpy as np
import pandas as pd
import logging
from data_generators.generator_util import get_join_root_dir_map
from data_generators.generator_util import load_image
from data_generators.generator_util import cache_file


logging.getLogger().setLevel(logging.INFO)
def create_dataset(df, config, transforms):
    output_shape = config["DATA_GENERATOR"]["OUTPUT_SHAPE"]
    output_shape = (eval(output_shape[0]), eval(output_shape[1]))
    output_image_channels = config["DATA_GENERATOR"]["OUTPUT_IMAGE_CHANNELS"]
    output_image_type = tf.dtypes.as_dtype(np.dtype(config["DATA_GENERATOR"]["OUTPUT_IMAGE_TYPE"]))
    data_dir = config["DATA_GENERATOR"]["DATA_DIR"]
    batch_size = config["BATCH_SIZE"]
    drop_remainder = eval(config["DATA_GENERATOR"]["DROP_REMAINDER"])
    cache_dir = config["DATA_GENERATOR"]["CACHE_DIR"]
    repeat = eval(config["DATA_GENERATOR"]["REPEAT"])
    num_parallel_calls = config["DATA_GENERATOR"]["NUM_PARALLEL_CALLS"]
    n_classes = config["NUM_CLASSES"]


    if isinstance(df, pd.DataFrame) == False:
        raise ValueError("ERROR: Dataset is not DataFrame. DataFrame is required")
    df = df.copy()
    df['image_path'] = df['image_path'].apply(get_join_root_dir_map(data_dir))
    df["label_name"] = df["label_name"].apply(str)
    df["label"] = df["label"].apply(multi_hot_encode, args=(n_classes,))
    dataset = tf.data.Dataset.from_tensor_slices(
        dict(image_path=df.image_path.values,
             label=np.array(list(df['label'].values))))
    dataset = dataset.map(load_data, num_parallel_calls=num_parallel_calls)
    dataset = dataset.cache(cache_file(cache_dir))
    if repeat is True:
        dataset = dataset.repeat()


    if transforms.transform["ImageTransform"] is not None:
        transforms_map = get_transform_map(transforms, output_shape, output_image_channels, output_image_type)
        dataset = dataset.map(transforms_map)
        logging.info(dataset)


    dataset = dataset.map(lambda row: ([row["image"], row['label']]))
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(4)
    return dataset


def get_transform_map(transforms, output_shape, output_image_channels, output_image_type):

    def new_transforms(image):
        row = transforms.transform["ImageTransform"](image=image)
        return row["image"]

    def transform_map(row):
        logging.info(tf.py_func(new_transforms, [row["image"]],[output_image_type]))
        image = tf.py_func(new_transforms, [row["image"]],
                           [output_image_type])[0]
        image.set_shape(output_shape + (output_image_channels,))
        row["image"] = image
        return row
    return transform_map


def load_data(row):
    image_path = row["image_path"]
    image = load_image(image_path)
    label = row["label"]
    new_row = dict(image=image, label=label)
    return new_row


def multi_hot_encode(label_indicies, n_classes):
    encoded = np.zeros((n_classes,))
    encoded[label_indicies] = 1
    return encoded