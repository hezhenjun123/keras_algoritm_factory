#!/usr/bin/env bash
export AWS_REGION="us-east-2"

RUN_ENV="$(sed -n '1p' $1 | cut -d' ' -f2)"
echo "RUN_ENV: $RUN_ENV"
RUN_MODE="$(sed -n '2p' $1 | cut -d' ' -f2)"
echo "RUN_MODE: $RUN_MODE"

if [ "$RUN_MODE" == "train" ] 
then
    SCRIPT="run_experiment.py"
elif [ "$RUN_MODE" == "inference" ] 
then
    SCRIPT="run_inference.py"
fi


if [ "$RUN_ENV" == "aws" ]; then
    pip install -r requirements.txt
fi

if ! [ -f utilities/bbox_overlap.c ]
then
    python utilities/bbox_setup.py build_ext --inplace && mv bbox_overlap.c* utilities/
else
    echo NO SETUP IS NEEDED
fi

python $SCRIPT --config $1