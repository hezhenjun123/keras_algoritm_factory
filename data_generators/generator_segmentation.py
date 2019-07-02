import os
import shutil
import tensorflow as tf
import logging
import numpy as np


logging.getLogger().setLevel(logging.INFO)
def dataset_from_dataframe(
    df,
    output_shape,
    output_image_channels,
    output_image_type,
    data_dir="./data",
    batch_size=1,
    drop_remainder=False,
    transforms=None,
    cache_dir=None,
    num_parallel_calls=4,
    repeat=False,
    for_keras_fit=False,
):
    """This functions takes a `pandas.DataFrame` containing `image_path`,
    `seg_label_path` and `label_names` columns object and returns a
    `tf.data.Dataset` object ready to use in the training loop.

    Parameters
    ----------
    df : pandas.DataFrame
        A `pandas.DataFrame` object containing `image_path`, `seg_label_path`
        and `label_names` columns.
    output_shape : tuple of int
        Two dimensional tuple representing height and width `(H, W)` of the
        images and labels after applying `transforms`.
    output_image_channels : int
        The number of output channels of an image after applying `transforms`.
    output_image_type : tf.dtype
        The dtype of the output image after applying `transforms`.
    data_dir : str, optional
        The root directory where the data is located. Default value is
        `"./data"`.
    batch_size : int, optional
        The size of the batches that the `tf.data.Dataset` object will provide.
        Default is `1`.
    drop_remainder : bool, optional
        Whether the last incomplete batch should be thrown away. Default is
        `False`.
    transforms : avi.images.transforms.Transform, optional
        A function that takes (image, label) both `numpy.ndarray` and
        transforms them. Default is `None`.
    training : bool, optional
        Whether this dataset is used for training, as opposed to using it for
        testing. Default is `True`.
    cache_dir : str, optional
        The directory where to cache the data. Caching the data speeds up the
        training if the data is in the cloud. The data will be cached before
        applying `transforms`.
    buffer_size : int, optional
        The size of the buffer to use for shuffling the dataset. Default is
        `100`.
    num_parallel_calls : int
        The number of parallel processes that run when applying `transforms` to
        the data. Default is `4`.
    repeat : bool, optional
        The dataset will repeat the data with `tf.data.Dataset.repeat`. Default
        is `False`.
    for_keras_fit : bool, optional
        Whether this dataset will be used with keras `tf.models.Model.fit`
        training. The dataset will only return `image` and `label` as a tuple.
        Default is `False`.

    Returns
    -------
    tf.data.Dataset
        A dataset ready to use with `TensorFlow`.
    """
    df = df.copy()
    df["seg_label_path"] = df["seg_label_path"].fillna("")
    df["seg_label_path"] = df["seg_label_path"].apply(
        get_join_root_dir_map(data_dir)
    )
    df["image_path"] = df["image_path"].apply(get_join_root_dir_map(data_dir))
    label_ext = df["seg_label_path"].map(lambda s: os.path.splitext(s)[1])
    df["label_names"] = df["label_names"].apply(str)
    dataset = tf.data.Dataset.from_tensor_slices(
        dict(
            image_path=df.image_path.values,
            seg_label_path=df.seg_label_path.values,
            name=df.image_path.apply(os.path.basename).values,
            defects=df.label_names.values,
            label_ext=label_ext,
        )
    )

    dataset = dataset.map(load_data, num_parallel_calls=num_parallel_calls)
    if cache_dir:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        dataset = dataset.cache(os.path.join(cache_dir, "cache"))

    if repeat:
        dataset = dataset.repeat()

    if transforms:
        transform_map = get_transform_map(transforms, output_shape, output_image_channels, output_image_type)
        dataset = dataset.map(transform_map)

    if for_keras_fit:
        dataset = dataset.map(
            lambda row: (row["image"], row["segmentation_labels"])
        )

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.prefetch(4)
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
        label = row["segmentation_labels"]
        label.set_shape(output_shape + (output_image_channels,))
        label = label[:, :, 2]
        label = tf.math.greater(tf.cast(label, tf.float32), 255*0.66)
        label = tf.cast(label, tf.int8)
        label = tf.expand_dims(label, -1)
        logging.info(label)
        row["image"] = image
        row["segmentation_labels"] = label
        return row
    return transform_map


def get_join_root_dir_map(root_dir):
    """This function builds a map that takes a `relative_path` and joins it
    with `root_dir`.

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


def load_data(row):
    image = load_image(row["image_path"])
    label = load_image(row["seg_label_path"])

    new_row = dict(
        image=image,
        segmentation_labels=label,
        name=row["name"],
        defects=row["defects"],
    )
    return new_row


def load_image(path):
    """Loads an image located in `path`.

    Parameters
    ----------
    path : Tensor of dtype tf.string.
        The full path to the label `png` file. It assumes the path is valid.

    Returns
    -------
    tf.Tensor
        A Tensor of type `tf.uint8` and shape `(H, W, C)`.
    """
    image = tf.read_file(path)
    image = tf.image.decode_image(image)
    return image

