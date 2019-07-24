import os
import tensorflow as tf
import numpy as np
import copy
import logging
from data_generators.generator_segmentation_vanilla import GeneratorSegmentationVanilla

logging.getLogger().setLevel(logging.INFO)


def get_plot_data(df, crop_and_expand, config):
    """Reads the images and segmentation labels listed in `dataframe` and
    located in the root dir `data_dir`, and returns them as a list of tuples
    with images (`numpy.ndarray`), segmentation labels (`numpy.ndarray`) and
    name (`str`).

    Parameters
    ----------
    df : pandas.DataFrame
        A `pandas.DataFrame` object containing `image_path`, `seg_label_path`
        and `label_names` columns.
    data_dir : str
        The root directory where the data is located.

    Returns
    -------
    list of tuple
        A list of tuples with the images (`numpy.ndarray`), segmentation labels
        (`numpy.ndarray`) and names (`str`) of the data listed in `df`.
    """
    plot_data = []
    config_now = copy.deepcopy(config)
    config_now["BATCH_SIZE"] = 1
    config_now["DATA_GENERATOR"]["OUTPUT_SHAPE"] = ["None", "None"]
    config_now["DATA_GENERATOR"]["OUTPUT_IMAGE_TYPE"] = "uint8"
    config_now["DATA_GENERATOR"]["REPEAT"] = False

    with tf.Graph().as_default(), tf.Session() as sess:
        segmentation_generator = GeneratorSegmentationVanilla(config_now)
        dataset = segmentation_generator.create_dataset_for_plot(df=df)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        while True:
            try:
                feat = sess.run(features)
                rnd_x = np.random.randint(0, 1536)
                rnd_y = np.random.randint(0, 2080)
                plot_data.append((
                    feat["image"][:, rnd_x:(rnd_x +
                                            crop_and_expand.resize[0]), rnd_y:(
                                                rnd_y +
                                                crop_and_expand.resize[1])],
                    feat["segmentation_labels"][:, rnd_x:(
                        rnd_x + crop_and_expand.resize[0]), rnd_y:(
                            rnd_y + crop_and_expand.resize[1])],
                    "",
                ))
            except tf.errors.OutOfRangeError:
                break
    return plot_data
