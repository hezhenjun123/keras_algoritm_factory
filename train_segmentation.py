import os
import tensorflow as tf
import pandas as pd
from math import ceil
import json
import argparse
import logging


from model.unet import UNet
from loss.dice import Dice
from metric.mean_iou import MeanIOU
from utilities.image_summary import ImageSummary
from utilities.color import generate_colormap
from utilities.crop_patches import CropAndExpand
from utilities.cos_anneal import CosineAnnealingScheduler
from utilities.smart_checkpoint import SmartCheckpoint
from utilities.helper import create_run_dir
from utilities.helper import get_plot_data
from data_generators.generator_segmentation import create_dataset
from data_generators.transforms import TransformUVSegmentation


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.logging.set_verbosity(tf.logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)


# tf.enable_eager_execution()
parser = argparse.ArgumentParser()
parser.add_argument("--run_env", type=str, required=True, choices=["aws", "local"])
parser.add_argument("--config", type=str, required=True)


def start_experiment(config, args):
    """Function to run an experiment training a UNet model with tensorflow.

    Parameters
    ----------
    batch_size : int, optional
        The size of the batch to train with, by default 1.
    data_csv : str, optional
        The path to the csv file with the training data. The file should have
        the columns `image_path`, `seg_label_path`, `label_names` and `split`.
        By default "data.csv".
    data_dir : str, optional
        The root dir where the data referenced in `data_csv` is located, by
        default "./data".
    channel_list : tuple, optional
        The list that defines the architecture of the U-Net model, by default
        (32, 64, 128, 256, 512).
    num_classes : int, optional
        The number of segmentation classes to segment, by default 2.
    activation : str, optional
        The activation function tu use in the U-Net architecture, by default
        "relu".
    save_dir : str, optional
        The path to the folder to save the training data and checkpoints to, by
        default "./output".
    steps_per_epoch : int, optional
        The number of steps that each epoch is defined to have, by default 700.
    num_plots : int, optional
        The number of image-segmentation plots to save to tensorboard, by
        default 10.
    create_run_folder : bool, optional
        Whether to create an enumerated unique `run{number}` folder inside
        `save_dir`, by default True.
    """
    batch_size = config["BATCH_SIZE"]
    data_csv = config["DATA_CSV"]
    learning_rate = config["LEARNING_RATE"]
    dropout = config["DROPOUT"]
    channel_list = config["CHANNEL_LIST"]
    num_classes = config["NUM_CLASSES"]
    activation = config["ACTIVATION"]
    epochs = config["EPOCHS"]
    steps_per_epoch = config["STEPS_PER_EPOCH"]
    resize_config = config["TRANSFORM"]["RESIZE"]
    resize = (resize_config[0], resize_config[1])
    num_plots = config["NUM_PLOTS"]
    data_dir = config["DATA_GENERATOR"]["DATA_DIR"]
    if args.run_env == "aws":
        save_dir = config["AWS_PARA"]["DIR_OUT"]
    elif args.run_env == "local":
        save_dir = create_run_dir(config["LOCAL_PARA"]["DIR_OUT"])
    else:
        raise Exception("Incorrect RUN_ENV: {}".format(config["RUN_ENV"]))


    boxes = [[x / 4, y / 4, (x + 1) / 4,
              min((y + 1) / 4, .95)] for x in range(4) for y in range(4)]
    crop_and_expand = CropAndExpand(resize=resize, boxes=boxes)
    train_transforms = TransformUVSegmentation(config)
    valid_transforms = TransformUVSegmentation(config)

    data_csv = pd.read_csv(data_csv, sep='\t').fillna("")
    logging.info(data_csv.head(5))
    logging.info("#" * 15 + "Reading training data" + "#" * 15)
    data_train = data_csv[data_csv["split"] == "train"].sample(frac=1)
    train_dataset = create_dataset(df=data_train, config=config, transforms=train_transforms)
    logging.info("========train_dataset========")
    logging.info(train_dataset)
    train_dataset = train_dataset. \
        apply(tf.data.experimental.unbatch()).\
        map(crop_and_expand, num_parallel_calls=4). \
        shuffle(buffer_size=100, seed=1). \
        apply(tf.data.experimental.unbatch()).\
        batch(batch_size).map(
            lambda row: (row["image"], row["segmentation_labels"])
        )

    logging.info("#" * 15 + "Reading test data" + "#" * 15)
    data_valid = data_csv[data_csv["split"] == "valid"].sample(frac=1)
    valid_dataset = create_dataset(df=data_valid, config=config, transforms=valid_transforms)

    valid_dataset = valid_dataset. \
        apply(tf.data.experimental.unbatch()).\
        map(crop_and_expand, num_parallel_calls=4). \
        shuffle(buffer_size=100, seed=1). \
        apply(tf.data.experimental.unbatch()).\
        batch(batch_size).map(
            lambda row: (row["image"], row["segmentation_labels"])
        )

    model = UNet(
        input_shape=(*resize, 3),
        channel_list=channel_list,
        num_classes=num_classes,
        return_logits=False,
        activation=activation,
        dropout_prob=dropout,
        dropout_type="spatial",
        name="unet",
        input_name="images",
        output_name="seg_map",
        conv_block="default",
        normalization="layernorm",
    )
    model.summary()
    if steps_per_epoch == -1:
        steps_per_epoch = ceil(len(data_train) * 16 / batch_size)

    valid_steps = ceil(len(data_valid) * 16 / batch_size)
    plot_df = data_valid.sample(n=num_plots, random_state=69)
    # plot_df = data_train
    data_to_plot = get_plot_data(plot_df, crop_and_expand, config)

    summaries_dir = os.path.join(save_dir, "summaries")
    checkpoints_dir = os.path.join(save_dir, "checkpoints/")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=Dice(),
        metrics=[MeanIOU(num_classes=num_classes)],
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=summaries_dir)

    if num_classes == 2:
        cmap = "viridis"
    else:
        cmap = generate_colormap(num_classes, "ADE20K")

    logging.info('STARTING TRAINING, {} train steps, {} valid steps'.format(
        steps_per_epoch, valid_steps))

    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=valid_steps,
        verbose=1,
        callbacks=[
            tensorboard_callback,
            ImageSummary(
                tensorboard_callback,
                data_to_plot,
                update_freq=10,
                transforms=valid_transforms,
                cmap=cmap,
            ),
            CosineAnnealingScheduler(20, learning_rate),
            SmartCheckpoint(destination_path=checkpoints_dir,
                            file_format='epoch_{epoch:04d}/cp.ckpt',
                            save_weights_only=False,
                            verbose=1,
                            monitor='val_mean_iou',
                            mode='max',
                            save_best_only=True),
        ],
    )


def main(args):
    MODULE_CONFIG_FILE = 'config/{}'.format(args.config)
    if os.path.exists(MODULE_CONFIG_FILE) is False:
        raise Exception("config file does not exist: {}".format(MODULE_CONFIG_FILE))
    with open(MODULE_CONFIG_FILE) as f:
        module_config_all = json.load(f)
    start_experiment(module_config_all, args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)