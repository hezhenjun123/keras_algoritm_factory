#!/bin/bash

python3 run_inference.py --config config/yield/model_config_regression_yield_absolute_newview2.yaml &

sleep 4m

python3 run_inference.py --config config/chaff/model_config_segmentation_chaff_2.yaml &

python3 run_inference.py --config config/lodging/model_config_segmentation_lodging_inference.yaml &

