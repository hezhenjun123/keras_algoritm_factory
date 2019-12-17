import copy
import logging
from data_generators.generator_factory import DataGeneratorFactory
import tensorflow as tf

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

    # with tf.Graph().as_default(), tf.Session() as sess:
    generator_factory = DataGeneratorFactory(config_now)
    generator = generator_factory.create_generator(config_now["EXPERIMENT"]["VALID_GENERATOR"])
    dataset = generator.create_dataset(df=df, transforms=None)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    while True:
        try:
            image, segmentation_labels = iterator.get_next()
            plot_data.append((
                image.numpy(),
                segmentation_labels.numpy(),
                "",
            ))
        except tf.errors.OutOfRangeError:
            break

    return plot_data

def get_tf_version():
    return int(tf.__version__.split('.')[0])

def allow_memory_growth():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

def fix_randomness(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if get_tf_version() <= 1:
        tf.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)

def config_gpu_memory(gpu_mem_cap):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return
    print('Found the following GPUs:')
    for gpu in gpus:
        print('  ', gpu)
    for gpu in gpus:
        try:
            if not gpu_mem_cap:
                tf.config.experimental.set_memory_growth(gpu, True)
            else:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=gpu_mem_cap)])
        except RuntimeError as e:
            print('Can not set GPU memory config', e)
