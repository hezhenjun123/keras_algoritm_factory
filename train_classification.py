import os
import data_generators.generator_classification as data
# import data_generators.data_classification as data
import tensorflow as tf
import pandas as pd
from math import ceil
from model.resnet import ResNet
import albumentations as A
from utilities.smart_checkpoint import SmartCheckpoint
import subprocess
import logging
import json
import argparse


logging.getLogger().setLevel(logging.INFO)
MODULE_CONFIG_FILE = 'config/model_config.json'

parser = argparse.ArgumentParser()
parser.add_argument("--run_env", type=str, required=True, choices=["aws", "local"])


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['S3_REQUEST_TIMEOUT_MSEC'] = '6000000'
tf.logging.set_verbosity(tf.logging.INFO)


def create_new_run_dir(save_dir):
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


def start_experiment(config, args):
    batch_size = config["BATCH_SIZE"]
    data_csv = config["DATA_CSV"]
    learning_rate = config["LEARNING_RATE"]
    pretrained = config["PRETRAINED"]
    dropout = config["DROPOUT"]
    channel_list = config["CHANNEL_LIST"]
    num_classes = config["NUM_CLASSES"]
    activation = config["ACTIVATION"]
    epochs = config["EPOCHS"]
    steps_per_epoch = config["STEPS_PER_EPOCH"]
    resize = config["RESIZE"]
    if args.run_env == "aws":
        save_dir = config["AWS_PARA"]["DIR_OUT"]
        data_dir = config["AWS_PARA"]["DATA_DIR"]
    elif args.run_env == "local":
        save_dir = create_new_run_dir(config["LOCAL_PARA"]["DIR_OUT"])
        data_dir = config["LOCAL_PARA"]["DATA_DIR"]

    else:
        raise Exception("Incorrect RUN_ENV: {}".format(config["RUN_ENV"]))


    train_transforms = A.Compose([A.RandomCrop(388, 388), A.Resize(resize[0], resize[1])])
    valid_transforms = A.Compose([A.RandomCrop(388, 388), A.Resize(resize[0], resize[1])])
    # train_transforms = None
    # valid_transforms = None
    data_csv = pd.read_parquet(data_csv).fillna("")
    print(data_csv.head())
    print("#" * 15 + "Reading training data" + "#" * 15)
    data_train = data_csv[data_csv["split"] == "train"].sample(frac=1)

    train_dataset = data.create_dataset(
        source=data_train,
        output_shape=(None, None),
        output_image_channels=3,
        output_image_type=tf.uint8,
        data_dir=data_dir,
        batch_size=batch_size,
        drop_remainder=False,
        transforms=train_transforms,
        training=True,
        cache_data=True,
        num_parallel_calls=4)

    print("#" * 15 + "Reading valid data" + "#" * 15)
    data_valid = data_csv[data_csv["split"] == "valid"].sample(frac=1)

    valid_dataset = data.create_dataset(
        source=data_valid,
        output_shape=(None, None),
        output_image_channels=3,
        output_image_type=tf.uint8,
        data_dir=data_dir,
        batch_size=batch_size,
        drop_remainder=False,
        transforms=valid_transforms,
        training=False,
        cache_data=True,
        num_parallel_calls=1)

    model = ResNet(input_shape=(*resize, 3),
                   pretrained=pretrained,
                   channel_list=channel_list,
                   num_classes=num_classes,
                   activation=activation,
                   dropout_prob=dropout,
                   name='unet',
                   input_name='images',
                   output_name='logits')
    model.summary()


    if steps_per_epoch == -1: steps_per_epoch = ceil(len(data_train) / batch_size)
    valid_steps = ceil(len(data_valid) / batch_size)

    summaries_dir = os.path.join(save_dir, "summaries")
    checkpoints_dir = os.path.join(save_dir, "checkpoints")

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=summaries_dir),
        SmartCheckpoint(checkpoints_dir, file_format='epoch_{epoch:04d}/cp.ckpt', period=10)
    ]
    model.fit(train_dataset,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=valid_steps,
              verbose=2,
              callbacks=callbacks)

    model.save("output/model_final.h5")
    model_save_path = os.path.join(save_dir, "model_saved")
    upload_command = 'aws s3 cp --recursive {} {}'.format("output", model_save_path)
    subprocess.check_call(upload_command.split(' '))


def main(args):
    with open(MODULE_CONFIG_FILE) as f:
        module_config_all = json.load(f)
    start_experiment(module_config_all, args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)