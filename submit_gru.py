from landingzone import gru

CONFIG_FILE = ["model_config_segmentation_chaff_tf1.yaml"]


def run_gru_job():
    for config in CONFIG_FILE:
        gru.submit(job_name='wheat_experiment',
                   project_name='wheat_yield',
                   job_decription=f'GRU RUN {config}',
                   cmd=f'run_experiment.sh {config}',
                   instance_type='p2.xlarge')


def main():
    run_gru_job()


if __name__ == "__main__":
    main()