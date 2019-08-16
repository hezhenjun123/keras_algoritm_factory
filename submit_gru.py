from landingzone import gru

CONFIG_FILE = ["model_config_segmentation_lodging.yaml"]


def run_gru_job():
    for config in CONFIG_FILE:
        gru.submit(job_name='wheat_lodging_segmentation',
                   project_name='wheat_yield',
                   job_decription=f'GRU RUN {config}',
                   cmd=f'run_experiment.sh {config}',
                   instance_type='p3.2xlarge',
                   timeout_hour=48)  # p2.xlarge, p3.2xlarge


def main():
    run_gru_job()


if __name__ == "__main__":
    main()
