import copy
import logging
from data_generators.generator_factory import  DataGeneratorFactory
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

logging.getLogger().setLevel(logging.INFO)


def get_plot_data(df, config):
    """Reads the images and segmentation labels listed in `dataframe` and
    located in the root dir `data_dir`, and returns them as a list of tuples
    with images (`numpy.ndarray`), segmentation labels (`numpy.ndarray`) and
    name (`str`).

    Parameters
    ----------
    df : pandas.DataFrame
        A `pandas.DataFrame` object containing `image_path`, `seg_label_path`
        and `label_names` columns.
    config : dict
        config dictionary

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
        generator_factory = DataGeneratorFactory(config_now)
        generator = generator_factory.create_generator(
            config_now["EXPERIMENT"]["VALID_GENERATOR"])
        dataset = generator.create_dataset(df=df, transforms=None)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        while True:
            try:
                image,segmentation_labels = sess.run(features)
                plot_data.append((
                    image,
                    segmentation_labels,
                    "",
                ))
            except tf.errors.OutOfRangeError:
                break
    return plot_data
