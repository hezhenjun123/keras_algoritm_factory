import tensorflow as tf
import numpy as np
import logging
from data_generators.generator_util import get_join_root_dir_map
from data_generators.generator_util import load_image
from data_generators.generator_util import cache_file


logging.getLogger().setLevel(logging.INFO)
def create_dataset(df, config, transforms=None):
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
    drop_remainder : bool, optional
        Whether the last incomplete batch should be thrown away. Default is
        `False`.
    transforms : avi.images.transforms.Transform, optional
        A function that takes (image, label) both `numpy.ndarray` and
        transforms them. Default is `None`.
    cache_dir : str, optional
        The directory where to cache the data. Caching the data speeds up the
        training if the data is in the cloud. The data will be cached before
        applying `transforms`.
    num_parallel_calls : int
        The number of parallel processes that run when applying `transforms` to
        the data. Default is `4`.
    repeat : bool, optional
        The dataset will repeat the data with `tf.data.Dataset.repeat`. Default
        is `False`.

    Returns
    -------
    tf.data.Dataset
        A dataset ready to use with `TensorFlow`.
    """

    output_shape = config["DATA_GENERATOR"]["OUTPUT_SHAPE"]
    output_shape = (output_shape[0], output_shape[1])
    output_image_channels = config["DATA_GENERATOR"]["OUTPUT_IMAGE_CHANNELS"]
    output_image_type = tf.dtypes.as_dtype(np.dtype(config["DATA_GENERATOR"]["OUTPUT_IMAGE_TYPE"]))
    data_dir = config["DATA_GENERATOR"]["DATA_DIR"]
    batch_size = config["BATCH_SIZE"]
    drop_remainder = config["DATA_GENERATOR"]["DROP_REMAINDER"]
    cache_dir = config["DATA_GENERATOR"]["CACHE_DIR"]
    repeat = config["DATA_GENERATOR"]["REPEAT"]
    num_parallel_calls = config["DATA_GENERATOR"]["NUM_PARALLEL_CALLS"]


    df = df.copy()
    df["segmentation_path"] = df["segmentation_path"].fillna("")
    df["segmentation_path"] = df["segmentation_path"].apply(
        get_join_root_dir_map(data_dir)
    )
    df["image_path"] = df["image_path"].apply(get_join_root_dir_map(data_dir))


    dataset = tf.data.Dataset.from_tensor_slices(
        dict(
            image_path=df.image_path.values,
            segmentation_path=df.segmentation_path.values
        )
    )
    dataset = dataset.map(load_data, num_parallel_calls=num_parallel_calls)
    dataset = dataset.cache(cache_file(cache_dir))
    if repeat is True:
        dataset = dataset.repeat()
    if transforms is not None and \
            (transforms.transform["ImageTransform"] is not None or \
            transforms.transform["ImageAndSegMapTransform"] is not None or \
            transforms.transform["SegMapTransform"] is not None):
        transform_map = get_transform_map(transforms, output_shape, output_image_channels, output_image_type)
        dataset = dataset.map(transform_map)
    # dataset = dataset.map(lambda row: (row["image"], row["segmentation_labels"]))
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(4)
    logging.info("==========================dataset=====================")
    logging.info(dataset)
    return dataset


def get_transform_map(transforms, output_shape, output_image_channels, output_image_type):

    def new_transforms(image, segmentation_labels):
        image = image
        seg_label = segmentation_labels
        if transforms.transform["ImageTransform"] is not None:
            image = transforms.transform["ImageTransform"](image=image)["image"]
        if transforms.transform["ImageAndSegMapTransform"] is not None:
            augmented = transforms.transform["ImageAndSegMapTransform"](image=image, mask=seg_label)
            image = augmented["image"]
            seg_label = augmented["mask"]
        if transforms.transform["SegMapTransform"] is not None:
            seg_label = transforms.transform["SegMapTransform"](mask=seg_label)["mask"]

        image = image.astype(output_image_type.name)
        seg_label = seg_label.astype(np.uint8)
        return [image, seg_label]


    def transform_map(row):
        logging.info(tf.py_func(new_transforms, [row["image"], row["segmentation_labels"]], \
                                [output_image_type, tf.uint8]))
        augmented = tf.py_func(new_transforms, [row["image"], row["segmentation_labels"]], \
                               [output_image_type, tf.uint8])
        image = augmented[0]
        image.set_shape(output_shape + (output_image_channels,))
        logging.info(image)
        label = augmented[1]


        label.set_shape(output_shape + (1,))
        logging.info(label)
        row["image"] = image
        row["segmentation_labels"] = label
        return row
    return transform_map


def load_data(row):
    image = load_image(row["image_path"])
    label = load_image(row["segmentation_path"])

    new_row = dict(
        image=image,
        segmentation_labels=label,
    )
    return new_row