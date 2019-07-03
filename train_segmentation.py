import os
import tensorflow as tf
import pandas as pd
from math import ceil
import albumentations as A
import json
import argparse


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
from data_generators.generator_segmentation import dataset_from_dataframe


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.logging.INFO)

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
    learning_rate : float, optional
        The learning rate to train the neural network with, by default 1E-4.
    dropout : float, optional
        The dropout to be used as regularization during training, by default
        0.25.
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
    epochs : int, optional
        The number of epochs to train for, by default 700.
    steps_per_epoch : int, optional
        The number of steps that each epoch is defined to have, by default 700.
    num_plots : int, optional
        The number of image-segmentation plots to save to tensorboard, by
        default 10.
    resize : tuple, optional
        A list in `[height, width]` format defining the size to resize the
        training images to, by default (388, 388).
    create_run_folder : bool, optional
        Whether to create an enumerated unique `run{number}` folder inside
        `save_dir`, by default True.
    """

    # TODO : USER INPUT.
    # Create the functions that transform the data. One for train data and one
    # for valid data. You can add data augmentation, resizing, etc. Check
    # `avi.exp.transforms` for more information on transforms.

    #make a set of boxes for crop and resize transform
    #crop of right side of image that has  reflection b


    batch_size = config["BATCH_SIZE"]
    data_csv = config["DATA_CSV"]
    learning_rate = config["LEARNING_RATE"]
    dropout = config["DROPOUT"]
    channel_list = config["CHANNEL_LIST"]
    num_classes = config["NUM_CLASSES"]
    activation = config["ACTIVATION"]
    epochs = config["EPOCHS"]
    steps_per_epoch = config["STEPS_PER_EPOCH"]
    resize = config["RESIZE"]
    num_plots = config["NUM_PLOTS"]
    if args.run_env == "aws":
        save_dir = config["AWS_PARA"]["DIR_OUT"]
        data_dir = config["AWS_PARA"]["DATA_DIR"]
    elif args.run_env == "local":
        save_dir = create_run_dir(config["LOCAL_PARA"]["DIR_OUT"])
        data_dir = config["LOCAL_PARA"]["DATA_DIR"]

    else:
        raise Exception("Incorrect RUN_ENV: {}".format(config["RUN_ENV"]))


    boxes = [[x / 4, y / 4, (x + 1) / 4,
              min((y + 1) / 4, .95)] for x in range(4) for y in range(4)]
    crop_and_expand = CropAndExpand(resize=resize, boxes=boxes)

    train_transforms = A.Compose([A.Normalize()])
    valid_transforms = A.Compose([A.Normalize()])
    # If you want to squash all the defects into just one kind of defect,
    # uncomment the next two lines.

    # train_transforms.transforms.append(tr.SquashSegmentationLabels())
    # valid_transforms.transforms.append(tr.SquashSegmentationLabels())

    data_csv = pd.read_csv(data_csv).fillna("")
    print(data_csv.head(5))
    print("#" * 15 + "Reading training data" + "#" * 15)
    data_train = data_csv[data_csv["split"] == "train"].sample(frac=1)
    train_dataset = dataset_from_dataframe(
        df=data_train,
        # TODO : USER INPUT.
        # If your transforms oputput a different output shape,
        # change output_shape and output_image_channels. If it
        # outputs a different type from tf.uint8, change
        # output_image_type.
        output_shape=(2048, 2592),
        output_image_channels=3,
        output_image_type=tf.float32,
        data_dir=data_dir,
        batch_size=batch_size,
        drop_remainder=False,
        transforms=train_transforms,
        cache_dir="./cache_train",
        num_parallel_calls=4,
        repeat=True,
        for_keras_fit=False,
    )

    train_dataset = train_dataset. \
        apply(tf.data.experimental.unbatch()).\
        map(crop_and_expand, num_parallel_calls=4). \
        shuffle(buffer_size=100, seed=1). \
        apply(tf.data.experimental.unbatch()).\
        batch(batch_size).map(
            lambda row: (row["image"], row["segmentation_labels"])
        )

    print("#" * 15 + "Reading test data" + "#" * 15)
    data_valid = data_csv[data_csv["split"] == "valid"].sample(frac=1)
    valid_dataset = dataset_from_dataframe(
        df=data_valid,
        # TODO : USER INPUT.
        # If your transforms oputput a different output shape,
        # change output_shape and output_image_channels. If it
        # outputs a different type from tf.uint8, change
        # output_image_type.
        output_shape=(2048, 2592),
        output_image_channels=3,
        output_image_type=tf.float32,
        data_dir=data_dir,
        batch_size=batch_size,
        drop_remainder=False,
        transforms=valid_transforms,
        cache_dir="./cache_valid",
        num_parallel_calls=4,
        repeat=True,
        for_keras_fit=False,
    )

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
    # Uncomment the following line to generate a png image with your model's
    # architecture.
    # tf.keras.utils.plot_model(model, show_shapes=True)
    if steps_per_epoch == -1:
        steps_per_epoch = ceil(len(data_train) * 16 / batch_size)

    valid_steps = ceil(len(data_valid) * 16 / batch_size)
    plot_df = data_valid.sample(n=num_plots, random_state=69)
    data_to_plot = get_plot_data(plot_df, data_dir, crop_and_expand)

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

    print('STARTING TRAINING, {} train steps, {} valid steps'.format(
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
                            file_format='epoch_{epoch:04d}/cp.h5',
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