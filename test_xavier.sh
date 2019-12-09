#!/bin/bash

python3 run_inference.py --config yield/model_config_regression_yield_absolute_newview2.yaml &
python3 run_inference.py --config chaff/model_config_segmentation_chaff_2.yaml &

#TODO: add the other three model
