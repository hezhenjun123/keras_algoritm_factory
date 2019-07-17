import os
import tensorflow as tf
import pandas as pd
from math import ceil
import json
import argparse
import logging


from utilities.smart_checkpoint import SmartCheckpoint
from data_generators.generator_classification import create_dataset
from data_generators.transforms import TransformSimpleClassiication
from model.resnet import ResNet


parser = argparse.ArgumentParser()
parser.add_argument("--run_env", type=str, required=True, choices=["aws", "local"])
parser.add_argument("--config", type=str, required=True)
logging.getLogger().setLevel(logging.INFO)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['S3_REQUEST_TIMEOUT_MSEC'] = '6000000'


def create_new_run_dir(save_dir):
    tf.gfile.MakeDirs(save_dir)
    list_of_files = tf.gfile.ListDirectory(save_dir)
    i = 1
    while f"run{i}" in list_of_files:
        i += 1
    run_dir = os.path.join(save_dir, f"run{i}")
    tf.gfile.MakeDirs(run_dir)
    logging.info("#" * 40)
    logging.info(f"Saving summaries on {run_dir}")
    logging.info("#" * 40)
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
    elif args.run_env == "local":
        save_dir = create_new_run_dir(config["LOCAL_PARA"]["DIR_OUT"])
    else:
        raise Exception("Incorrect RUN_ENV: {}".format(config["RUN_ENV"]))


    train_transforms = TransformSimpleClassiication(config)
    valid_transforms = TransformSimpleClassiication(config)
    data_csv = pd.read_csv(data_csv, sep='\t').fillna("")
    logging.info(data_csv.head())
    logging.info("#" * 15 + "Reading training data" + "#" * 15)
    data_train = data_csv[data_csv["split"] == "train"].sample(frac=1)
    train_dataset = create_dataset(df=data_train, config=config, transforms=train_transforms)
    logging.info("#" * 15 + "Reading valid data" + "#" * 15)
    data_valid = data_csv[data_csv["split"] == "valid"].sample(frac=1)

    valid_dataset = create_dataset(df=data_train, config=config, transforms=valid_transforms)

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