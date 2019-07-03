#Submit job to GRU

cd zoomlion-model-pipeline 

lzone-cli submit_job vibration-data wheat_yield --cmd="run_classification.sh --aws" --job_description='test new package'

lzone-cli submit_job vibration-data wheat_yield --cmd="run_segmentation.sh --aws" --job_description='test new package'
