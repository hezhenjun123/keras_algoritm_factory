import os
import tensorflow as tf
from datetime import datetime


def get_join_root_dir_map(root_dir):
    """This function builds a map that takes a `relative_path` and joins it with
    `root_dir`.
    """

    def join_root_dir_map(relative_path):
        total_path = ""
        if relative_path:
            total_path = os.path.join(root_dir, relative_path)
        return total_path

    return join_root_dir_map


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


def cache_file(cache_dir):
    if os.path.exists(cache_dir) is False:
        os.makedirs(cache_dir)
    curr_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cache_file = os.path.join(cache_dir, f"cache_{curr_time}")
    return cache_file
