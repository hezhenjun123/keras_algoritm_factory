#!/bin/bash


sudo python3 tegrastats2.py --bin=/usr/bin/tegrastats --output=./a.log --params "--interval 500"

sleep 1m

python3 run_inference.py --config config/yield/model_config_regression_yield_absolute_newview2.yaml &

sleep 4m

python3 run_inference.py --config config/chaff/model_config_segmentation_chaff_2.yaml &

sleep 5s

python3 run_inference.py --config config/lodging/model_config_segmentation_lodging_inference.yaml &

sleep 1m
python3 visualize.py --input="./3-model.log" --output="./3-model.xls"
