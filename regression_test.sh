#!/bin/bash

python --version
bash /opt/tensorrt/python/python_setup.sh
python utilities/bbox_setup.py build_ext --inplace && mv bbox_overlap.c* utilities/ && ls -lth
python run_inference.py --config=config/yield/model_config_regression_yield_absolute_newview2.yaml --freeze_to_pb_path ./tmp_resnet_model
python run_inference.py --config=config/yield/model_config_regression_yield_absolute_newview2.yaml --create_trt_engine --debug
