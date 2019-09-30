from landingzone import gru
from utilities.config import read_config
import subprocess
import argparse
from typing import List


CONFIG_FILES = ['model_config_segmentation_lodging.yaml']

def run_gru_job(config_path,gru_settings):
    gru.submit(job_name=gru_settings["JOB_NAME"],
               project_name=gru_settings["PROJECT_NAME"],
               job_decription=f'GRU RUN {config_path}',
               cmd=f'run.sh {config_path}',
               instance_type=gru_settings["INSTANCE_TYPE"],
               timeout_hour=gru_settings["TIMEOUT"])  # p2.xlarge, p3.2xlarge


def main(config_files):
    for config_path in config_files:
        config = read_config(config_path)
        if config["RUN_ENV"] == 'aws':
            run_gru_job(config_path,config["GRU"])
        else:
            subprocess.call(f"./run.sh {config_path}".split(' '))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    if args.config:
        main([args.config])
    else:
        main(CONFIG_FILES)
