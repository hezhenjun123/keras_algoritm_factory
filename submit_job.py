from landingzone import gru
from utilities.config import read_config
import subprocess

CONFIG_FILE = ["model_config_segmentation_breakage.yaml"]


def run_gru_job(config_path):
    gru.submit(job_name='wheat_breakage_segmentation',
               project_name='wheat_yield',
               job_decription=f'GRU RUN {config_path}',
               cmd=f'run.sh {config_path}',
               instance_type='p3.2xlarge',
               timeout_hour=96)  # p2.xlarge, p3.2xlarge


def main():
    for config_path in CONFIG_FILE:
        config = read_config(config_path)
        if config["RUN_ENV"] == 'aws':
            run_gru_job(config_path)
        else:
            subprocess.call(f"./run.sh {config_path}".split(' '))


if __name__ == "__main__":
    main()
