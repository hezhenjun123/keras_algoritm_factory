import json
import numpy as np


def generate_colormap(n, colormap_name):
    """This function generates an array of `n` RGB values that are
    distinguishable and user friendly. Usually to be used to plot segmentation
    data.

    Parameters
    ----------
    n : int
        The number of RGB values that the output list should have.
    colormap_name : str
        The name of the colormap to produce. The colormaps are read from the
        `colormap.json` file, so the `colormap_name` should exist there. Some
        available names are: `ADE20K`, `CITYSCAPES` and `MAPILLARY_VISTAS`.

    Returns
    -------
    np.ndarray
        An array of shape `(n, 3)` with `n` RGB values that correspond to the
        selected `colormap_name`.
    """
    with open("colormap.json") as jf:
        colormaps = json.load(jf)
    ade20k_cm = colormaps[colormap_name]
    return np.array(ade20k_cm["RGBPoints"][:n])
