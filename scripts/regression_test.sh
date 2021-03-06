#!/bin/bash

bash /opt/tensorrt/python/python_setup.sh
python3 utilities/bbox_setup.py build_ext --inplace && mv bbox_overlap.c* utilities/ && ls -lth

STAGE=0

function usage
{
    echo "usage: ./build_opencv.sh [[-s sourcedir ] | [-h]]"
    echo "-s | --stage choose which test to run (1, 2, 3...)"
    echo "-h | --help  This message"
}


while [ "$1" != "" ]; do
    case $1 in
        -s | --stage )      shift
			    STAGE=$1
                            ;;
	-h | --help )       shift
			    ;;
	* )                 usage
                            exit 1
    esac
    shift
done


case $STAGE in
    1)# trt yield model
	python3 run_inference.py --config config/prod/model_config_regression_yield_absolute_newview2.yaml --inference_engine  INFERENCE_TRT --freeze_to_pb_path ~/zoomlion-sample/tmp_resnet_model --upload && \
	    python3 run_inference.py --config config/prod/model_config_regression_yield_absolute_newview2.yaml --inference_engine INFERENCE_TRT --create_trt_engine --debug 
	;;
    2)# trt chaff hopper
	python3 run_inference.py --config config/prod/model_config_segmentation_chaff_hopper.yaml --inference_engine INFERENCE_TRT --freeze_to_pb_path ~/zoomlion-sample/tmp_unet_chaff_hopper_model --upload && \
	    python3 run_inference.py --config config/prod/model_config_segmentation_chaff_hopper.yaml --inference_engine INFERENCE_TRT --create_trt_engine --debug
	;;

    3)# trt lodging
	python3 run_inference.py --config config/prod/model_config_segmentation_lodging.yaml --inference_engine INFERENCE_TRT --freeze_to_pb_path ~/zoomlion-sample/tmp_unet_lodging_model --upload && \
	    python3 run_inference.py --config config/prod/model_config_segmentation_lodging.yaml --inference_engine INFERENCE_TRT --create_trt_engine --debug
	;;

    4)# trt chaff elevator
	python3 run_inference.py --config config/prod/model_config_segmentation_chaff_elevator.yaml --inference_engine INFERENCE_TRT --freeze_to_pb_path ~/zoomlion-sample/tmp_unet_chaff_elevator_model --upload && \
	    python3 run_inference.py --config config/prod/model_config_segmentation_chaff_elevator.yaml --inference_engine INFERENCE_TRT --create_trt_engine --debug
	;;

    5)# load trt test
	python3 run_inference.py --load_test_config ./config/prod/4_trt_models.yaml --inference_engine INFERENCE_TRT --create_trt_engine
	;;

    6)# load trtbtest fp16
	python3 run_inference.py --load_test_config ./config/prod/4_trt_models.yaml --inference_engine INFERENCE_TRT --create_trt_engine --fp_16
	;;

	7)# TF lodging model
	python3 run_inference.py --config config/prod/model_config_segmentation_lodging.yaml --inference_engine INFERENCE_TF
	;;

	8)# TF chaff hopper model
	python3 run_inference.py --config config/prod/model_config_segmentation_chaff_hopper.yaml --inference_engine INFERENCE_TF
	;;

	9)# TF chaff elevator model
	python3 run_inference.py --config config/prod/model_config_segmentation_chaff_elevator.yaml --inference_engine INFERENCE_TF
	;;

	10)# TF yield model
	python3 run_inference.py --config config/prod/model_config_regression_yield_absolute_newview2.yaml --inference_engine INFERENCE_TF
	;;

    *)
	echo -n "unknow"
esac
