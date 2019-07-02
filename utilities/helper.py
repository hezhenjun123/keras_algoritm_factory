import os
import tensorflow as tf
from data_generators.generator_segmentation import dataset_from_dataframe
import numpy as np


def create_run_dir(save_dir):
    """Creates a numbered directory named "run1". If directory "run1" already
    exists then creates directory "run2", and so on.

    Parameters
    ----------
    save_dir : str
        The root directory where to create the "run{number}" folder.

    Returns
    -------
    str
        The full path of the newly created "run{number}" folder.
    """
    tf.gfile.MakeDirs(save_dir)
    list_of_files = tf.gfile.ListDirectory(save_dir)
    i = 1
    while f"run{i}" in list_of_files:
        i += 1
    run_dir = os.path.join(save_dir, f"run{i}")
    tf.gfile.MakeDirs(run_dir)
    print("#" * 40)
    print(f"Saving summaries on {run_dir}")
    print("#" * 40)
    return run_dir


def get_plot_data(df, data_dir, crop_and_expand):
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
    with tf.Graph().as_default(), tf.Session() as sess:
        dataset = dataset_from_dataframe(
            df=df,
            output_shape=((None, None)),
            output_image_channels=3,
            output_image_type=tf.uint8,
            data_dir=data_dir,
            batch_size=1,
            drop_remainder=False,
            transforms=None,
            cache_dir=None,
            for_keras_fit=False,
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        while True:
            try:
                feat = sess.run(features)
                rnd_x = np.random.randint(0, 1536)
                rnd_y = np.random.randint(0, 2080)
                plot_data.append(
                    (
                        feat["image"][:, rnd_x:(rnd_x + crop_and_expand.resize[0]),
                        rnd_y:(rnd_y + crop_and_expand.resize[1])],
                        feat["segmentation_labels"][:, rnd_x:(rnd_x + crop_and_expand.resize[0]),
                        rnd_y:(rnd_y + crop_and_expand.resize[1])],
                        feat["name"][0].decode(),
                    )
                )
            except tf.errors.OutOfRangeError:
                break
    return plot_data