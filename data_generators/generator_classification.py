import os
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
logging.getLogger().setLevel(logging.INFO)


def create_dataset(source,
                    output_shape,
                    output_image_channels,
                    output_image_type,
                    data_dir="./data",
                    batch_size=1,
                    drop_remainder=False,
                    transforms=None,
                    training=True,
                    cache_data=True,
                    num_parallel_calls=4):


    if isinstance(source, pd.DataFrame) == False:
        raise ValueError("ERROR: Dataset is not DataFrame. DataFrame is required")
    df = source
    df = df.copy()
    df['image_path'] = df['image_path'].apply(get_join_root_dir_map(data_dir))
    df["label_names"] = df["label_names"].apply(str)
    n_classes = max(map(max, df["labels"].values)) + 1
    df["labels"] = df["labels"].apply(multi_hot_encode, args=(n_classes,))
    dataset = tf.data.Dataset.from_tensor_slices(
        dict(image_path=df.image_path.values,
             label=np.array(list(df['labels'].values))))
    dataset = dataset.map(load_image, num_parallel_calls=num_parallel_calls)

    if cache_data:
        if training:
            cache_dir = "./cache"
        else:
            cache_dir = "./cache_valid"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        dataset = dataset.cache(os.path.join(cache_dir, "cache"))

    dataset = dataset.repeat()

    if transforms:
        transforms_map = get_transform_map(transforms, output_shape, output_image_channels, output_image_type)
        dataset = dataset.map(transforms_map)
        logging.info(dataset)


    dataset = dataset.map(
        lambda row: ([row["image"], row['label']])
    )

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(4)
    logging.info(dataset)
    return dataset


def get_transform_map(transforms, output_shape, output_image_channels, output_image_type):

    def new_transforms(image):
        row = transforms(image=image)
        return row["image"]

    def transform_map(row):
        logging.info(tf.py_func(new_transforms, [row["image"]],[output_image_type]))
        image = tf.py_func(new_transforms, [row["image"]],
                           [output_image_type])[0]
        image.set_shape(output_shape + (output_image_channels,))
        row["image"] = image
        return row
    return transform_map


def get_join_root_dir_map(root_dir):
    """This function builds a map that takes a `relative_path` and joins it with
    `root_dir`.

    Parameters
    ----------
    root_dir : str
        The path to the root directory.

    Returns
    -------
    function
        Function that takes a `relative_path` and joins it with `root_dir`.
    """

    def join_root_dir_map(relative_path):
        total_path = ""
        if relative_path:
            total_path = os.path.join(root_dir, relative_path)
        return total_path

    return join_root_dir_map


def load_image(row):
    """Takes a dictionary of `tf.Tensors` that contains features `image_path`, `label_path`,
    `name` and `defects`, loads the data in `image_path` and `label_path` and converts
    the dictionary into a dictionary of `tf.Tensors` containing `image`, `label`, `name`
    and `defects`.

    Parameters
    ----------
    row : dict of tf.Tensor
        An example coming from the `tf.data.Dataset` object. It should have keys `image_path`,
        `label_path`, `name` and `defects`.

    Returns
    -------
    dict of tf.Tensor
        A dictionary containing one data example with `image`, `label`, `name` and `defects`.
    """
    image_path = row["image_path"]
    image = tf.read_file(image_path)
    image = tf.image.decode_image(image)
    label = row["label"]
    new_row = dict(image=image, label=label)

    return new_row


def file_exists(file_path):
    """Takes a tf.string tensor representing the path to a file and returns
    a tf.bool tensor indicating whether the file exists.

    Parameters
    ----------
    file_path : tf.Tensor
        A string tensor representing the path to a file.
    """
    exists = tf.py_func(tf.gfile.Exists, [file_path], tf.bool)
    exists.set_shape([])
    return exists


def multi_hot_encode(label_indicies, n_classes):
    encoded = np.zeros((n_classes,))
    encoded[label_indicies] = 1
    return encoded