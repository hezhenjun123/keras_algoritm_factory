#!/usr/bin/env bash
export AWS_REGION="us-east-2"

RUN_ENV="$(sed -n '1p' config/$1 | cut -d' ' -f2)"
echo "RUN_ENV: $RUN_ENV"
RUN_MODE="$(sed -n '2p' config/$1 | cut -d' ' -f2)"
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
python $SCRIPT --config $1