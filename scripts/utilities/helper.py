import copy
import logging
from data_generators.generator_factory import DataGeneratorFactory
import tensorflow as tf
import os
import cv2
import datetime

logging.getLogger().setLevel(logging.INFO)

class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._num_frames += 1

    def elapsed(self):
        return (self._end - self._start).total_seconds()


    def fps(self):
        return self._num_frames * 1.0 / (datetime.datetime.now() - self._start).total_seconds()

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

def stream_video(vid_name, fps=None, imscale=1):
    """ Args:
        video_name (str): local filename or webcam ID
        fps (int): optionally downsample video by skipping frames.
            If video is saved at 30 fps, you can choose to load only 3 fps
        imscale (int): optionally downscale image to a lower dimension
    """

    logging.info("loading %s ... ", vid_name)

    if vid_name.isnumeric():
        cap = cv2.VideoCapture(int(vid_name))
    elif os.path.exists(vid_name):
        cap = cv2.VideoCapture(vid_name)
    else:
        logging.info("cannot find file %s", vid_name)
        raise FileNotFoundError(vid_name)

    act_fps = cap.get(cv2.CAP_PROP_FPS)
    act_fps = 30 if act_fps > 100 else act_fps
    fps = act_fps if fps is None else fps

    ratio = act_fps // fps
    logging.info("fps: %d, act_fps %d, ratio: %.3f", fps, act_fps, ratio)
    assert fps <= act_fps and ratio >= 1

    idx = -1
    r_fps = FPS().start()

    while cap.isOpened():
        idx += 1
        ret, frame = cap.read()

        r_fps.update()

        if not ret:
            break

        if idx % 200 == 0:
            logging.info("video loading fps: %.3f", r_fps.fps())

        if idx % ratio != 0:
            continue

        H, W, C = frame.shape
        Wz, Hz = W // imscale, H // imscale
        sframe = cv2.resize(frame, (Wz, Hz))

        yield sframe

    cap.release()
