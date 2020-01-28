#!/bin/bash

python --version
bash /opt/tensorrt/python/python_setup.sh
python utilities/bbox_setup.py build_ext --inplace && mv bbox_overlap.c* utilities/ && ls -lth


# yield model
python run_inference.py --config config/yield/model_config_regression_yield_absolute_newview2.yaml --freeze_to_pb_path ./tmp_resnet_model
python run_inference.py --config config/yield/model_config_regression_yield_absolute_newview2.yaml --create_trt_engine --debug

# chaff hopper
python run_inference.py --config config/yield/model_config_regression_yield_absolute_newview2.yaml --freeze_to_pb_path ./tmp_unet_chaff_hopper_model
python run_inference.py --config config/chaff_hopper/model_config_segmentation_chaff.yaml --create_trt_engine --debug

# lodging
python run_inference.py --config config/lodging/model_config_segmentation_lodging.yaml --freeze_to_pb_path ./tmp_unet_lodging_model
python run_inference.py --config config/lodging/model_config_segmentation_lodging.yaml --create_trt_engine --debug

# chaff elevator
python run_inference.py --config config/chaff_elevator/model_config_segmentation_chaff.yaml --freeze_to_pb_path ./tmp_unet_chaff_elevator_model
python run_inference.py --config config/chaff_elevator/model_config_segmentation_chaff.yaml --create_trt_engine --debug
