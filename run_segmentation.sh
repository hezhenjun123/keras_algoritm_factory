#!/usr/bin/env bash


AWS_REGION="us-east-2"
CLOUD_CMD="\
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
AWS_REGION=$AWS_REGION \
"


TASK='train_segmentation.py'
RUN_ENV="unknown"
for var in "$@"
do
    # Local and floyd IO
    if [ $var == "--local" ]; then
        RUN_ENV="local"
    elif [ $var == "--aws" ]; then
        RUN_ENV="aws"
    fi
done


# Python command
PY_CMD="python $TASK"
echo $PY_CMD
if [ "$RUN_ENV" == "local" ]; then
    $PY_CMD --run_env "local" --config model_config_segmentation.json
elif [ "$RUN_ENV" == "aws" ]; then
    pip install -r requirements.txt
    $PY_CMD --run_env "aws" --config model_config_segmentation.json
else
    echo "The job could not be started because there was no engine especified. \
    Please run with either of the follwing engines: --local , --aws"
fi