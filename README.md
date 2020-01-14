# Running a config

The command below is the entry point to the pipeline. 
If config file has `RUN_ENV: local`, the script will run locally, and if `RUN_ENV: aws`, 
it will package the current  directory and submit it as a job to GRU.

```
$ cd zoomlion-model-pipeline
$ python submit_job.py 


$ python run_experiment.py --config breakage/model_config_segmentation_breakage.yaml
$ python run_experiment.py --config chaff/model_config_segmentation_chaff.yaml
$ python run_experiment.py --config lodging/model_config_segmentation_lodging.yaml
$ python run_experiment.py --config yield/model_config_regression_yield_absolute.yaml
```

# Setup
AMI id: ami-0821a34583de81474



# Run inference on xavier

* Follow landing-eye repo to flash the miivii box

* Install dependencies
```
sudo python3 -m pip install -r requirements-xavier.txt
```

* Run inference

```

```

# Debug

###  fatal error: 'numpy/arrayobject.h' file not found

You may get the error below while running ./run.sh in a venv

```bash
(venv) ~/zoomlion-model-pipeline (clef-sprayer-weed)$ ./run.sh model_config_segmentation_sprayerweed.yaml 
Running config/model_config_segmentation_sprayerweed.yaml
RUN_ENV: local
RUN_MODE: train
Compiling utilities/bbox_overlap.pyx because it changed.
[1/1] Cythonizing utilities/bbox_overlap.pyx
running build_ext
building 'bbox_overlap' extension
clang -DNDEBUG -g -fwrapv -O3 -Wall -I/usr/local/include -I/usr/local/opt/openssl/include -I/usr/local/opt/sqlite/include -I/Users/suhabebugrara/zoomlion-model-pipeline/venv/include -I/usr/local/Cellar/python/3.7.3/Frameworks/Python.framework/Versions/3.7/include/python3.7m -c utilities/bbox_overlap.c -o build/temp.macosx-10.13-x86_64-3.7/utilities/bbox_overlap.o
utilities/bbox_overlap.c:611:10: fatal error: 'numpy/arrayobject.h' file not found
#include "numpy/arrayobject.h"
         ^~~~~~~~~~~~~~~~~~~~~
1 error generated.

```

One workaround is to set the following environment variable:
```
export CFLAGS="-I ./venv/lib/python3.7/site-packages/numpy/core/include $CFLAGS"
```
