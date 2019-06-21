<p align="center"><img width="60%" src="docs/img/avi-common-logo.png"/></p> 

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/landing-ai/avi-common/issues)

---

The AVI Common repository is the go-to code base when you want to jump-start your segmentation task (classiffication task
with more work needed).

# FOR TRANSFORMERS (TO BE DELETED IF MERGING INTO OTHER BRANCHES) 

## Modified branch that includes ability to run classificaiton tasks using Resnet50 architecture 


Read cloning section of the readme. Checkout branch keras-classification-modules in  `avi/` that contains minor modification to core `avi` library
To run classificaiton experiment you need to modify execute.sh with appropriate settings and experiments/experiment_classification.py if you want to modify Transforms or checkpointing

### Data File

This file defines s3 paths to images  and their respective labels. Instead of CSVs as in segmentation this files needs to be in parquet format that preserves all of the objects in dataframe in their original format
You should create your own for each task. For cv-yield monitoring there is a small script under data_generators/create_dfs.py that parses the data locally to create this dataframe 

### HOW TO RUN 

When appropriate modification are done in execute.sh and experiments/experiment_classification.py you can execute training as:

`./execute.sh --local` to run locally (you should do `pip install -r requirements.txt && pip install -e avi --no-deps` if running for the first time)

`lzone-cli submit_job JOB_NAME_OF_YOUR_CHOOSING wheat_yield --cmd="execute.sh --aws" --job_description='Your description here'` for GRU run. You need to follow [lzone-cli installation](https://landingai.atlassian.net/wiki/spaces/EN/pages/318275688/Gru+User+Guide) first



# Original README
## Code structure

The `data.py` module let's you create the `tf.data.Dataset` object needed for training. This
object can be created either from a `TFRecords` file or a `csv` file with columns `image_path`,
`seg_label_path` and `label_names`. The `to_tfrecords.py` script helps you transform images and labels 
data into a `TFRecords` file. The `experiment.py` file is the main script for training and this is the
one that you should probably modify. The `execute.sh` shell script is a script that helps you train the
model on FloydHub or locally quickly. So you probably want to modify this too. In this two files, you will find
`# TODO: USER INPUT` comments in the places that people most commonly need to modify when starting a new project.
The rest of the files are the training loop and the helper functions to train the segmentation task with a `UNet` model.

## Cloning

To clone the repository first clone and install https://github.com/landing-ai/avi locally and then use:

```bash
$ git clone --recurse-submodules git@github.com:landing-ai/avi-common-tf.git
```

After cloning, and especially if you are going to use FloydHub, remove the `tensorflow>=1.11` line from the `requirements.txt`
file inside the `avi` submodule in the repository. This line causes some parts of `TensorFlow` to reinstall and crashes training.

## Generate a TFRecords file

Training with `TFRecords` can give you a speed performance increase when training your model.

If you want to train a `UNet` model with `TFRecords` files, you should generate the file first. The `to_tfrecords.py` script is designed to do so as follows:

```bash
$ python to_tfrecords.py --csv_path data_csv/train.csv --data_dir data/ --output_dir tfrecords/train/ --num_shards 3
```

The previous command works when the `csv` file with the paths to the data is in `data_csv/train.csv`, you want to save the output file to `tfrecords/train/` directory, the root dir of your data is `data/` nd you want to split the data into `3` different `TFRecords` files.

The `csv` file should have the following columns: `image_path`, `label_names` and `seg_label_path`. The `image_path` is the path to
to the image, the `label_names` are the names of the labels in the image, and the `seg_label_path` is the path to the `.npz` or `.png` file with the pixel-wise corresponding label.

## Train the UNet model

Once you have a `TFRecords` file or if you want to train directly with the `csv` file with the paths to the images and labels 
(in `png` or `npz` format), you can train the model with the `experiment.py` python script. Be sure to modify the lines of 
this script where there are `# TODO: USER INPUT.` comments. Especially the preprocessing steps. This script has the following flags:

```bash
       USAGE: experiment.py [flags]
flags:

experiment.py:
  --activation: <relu|elu>: The activation function to use: relu or elu.
    (default: 'relu')
  --batch_size: The size of the batches for training.
    (default: '1')
    (an integer)
  --[no]batchnorm: Use batchnorm.
    (default: 'false')
  --channel_list: The channels architecture of UNet.
    (default: '[64,128,256,512,1024]')
  --data_dir: The root directory where the data is located.
    (default: './data')
  --dropout: Dropout probability.
    (default: '0.5')
    (a number)
  --epochs: The number of epochs to train the model.
    (default: '200')
    (an integer)
  --[no]groupnorm: Use groupnorm.
    (default: 'false')
  --groups: The number of groups for groupnorm.
    (default: '4')
    (an integer)
  --learning_rate: Learning rate.
    (default: '0.0001')
    (a number)
  --n_plots: The number of sample images to plot every `plot_int` epochs.
    (default: '30')
    (an integer)
  --plot_int: The number of epochs between saving of plots.
    (default: '20')
    (an integer)
  --resize: The height and width to resize the images to.
    (default: '[388, 388]')
  --save_dir: The directory to save the outputs of training.
    (default: 'output/')
  --[no]squashed: Use squashed dataset.
    (default: 'false')
  --valid_csv: The path to the csv with the valid images and labels paths.
  --valid_tfrecords: The path or list of paths to the valid tfrecords files.
  --train_csv: The path to the csv with the train images and labels paths.
  --train_tfrecords: The path or list of paths to the train tfrecords files.
  --weights: The per-class weights for the loss function.
    (default: '[1,1]')
```

You could also modify and use the `execute.sh` shell script to train your model either locally or on FloydHub. Remember to give the script the right permissions to execute with:

```bash
$ chmod +x execute.sh
```

## Perform inference with a TensorFlow SavedModel

To perfrom inference after you have already trained a model, you will also need either a `csv` or  `TFRecords` files. You will perform inference with the `segment.py` script. Be sure to modify the lines of this script where there are `# TODO: USER INPUT.` comments. Especially the preprocessing steps. This script can be run with the following flags:

```bash
       USAGE: segment.py [flags]
flags:

segment.py:
  --batch_size: The size of the batches for inference.
    (default: '1')
    (an integer)
  --csv: The .csv file listing the images to run inference on.
  --data_dir: The root directory where data is stored.
  --model_dir: The path to the TF saved model.
  --out_dir: The directory to save the segmentation results to.
  --tfrecords: The TFRecords file to run inference on.
```

You could also modify and use the `inference.sh` shell script to run iinference either locally or on FloydHub. Remember to give the script the right permissions to execute with:

```bash
$ chmod +x inference.sh
```

## Deploy Model

To deploy the model you can check out the [avi-model-tools](https://github.com/landing-ai/avi-model-tools) repository. This repo contains instructions on how to freeze the model to create a saved model protobuf file for deployment and also scripts to do this.
