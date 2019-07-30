#!/usr/bin/env bash
export AWS_REGION="us-east-2"

RUN_ENV="$(head -n 1 config/$1 | cut -d' ' -f2)"
echo "RUN_ENV: $RUN_ENV"
if [ "$RUN_ENV" == "aws" ]; then
    pip install -r requirements.txt
fi
# the arguments is the config file name
python run_experiment.py --config $1