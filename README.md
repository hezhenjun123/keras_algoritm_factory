#Submit job to GRU
cd zoomlion-model-pipeline 
python submit_gru.py

#Run locally
#change the first line of config to "local"
run_experiment.sh {config}
Example: "./run_experiment.sh model_config_classification.yaml"
