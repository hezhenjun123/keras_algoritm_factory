FROM 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-dev
 
WORKDIR /
RUN apt update && apt install -y --fix-missing make g++
RUN apt update && apt install -y --fix-missing python3-pip libhdf5-serial-dev hdf5-tools
RUN apt update && apt install -y python3-h5py

RUN apt-get install -y libatlas3-base \
 && apt-get install -y git \
 && apt-get install -y git-lfs && git-lfs install

RUN apt-get update && apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev

RUN apt-get install -y python3-pip
RUN pip3 install -U pip testresources setuptools
RUN pip3 install -U numpy==1.16.1 future==0.17.1 mock==3.0.5

RUN python3 -m pip install absl-py==0.7.1 \
astor==0.8.0 \
gast==0.2.2 \
google-pasta==0.1.7 \
h5py==2.9.0 \
Keras-Applications==1.0.8 \
Keras-Preprocessing==1.1.0 \
Markdown==3.1.1 \
six==1.12.0 \
termcolor==1.1.0 \
Werkzeug==0.15.4 \
wrapt==1.11.2 \
xmltodict

RUN pip3 install keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 enum34 futures protobuf
RUN python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.14.0

RUN python3 -m pip install cython
RUN python3 -m pip install pillow
RUN apt-get update
RUN apt-get install -y libopenblas-dev gfortran
RUN python3 -m pip uninstall -y enum34
RUN pip3 install pybind11
RUN pip3 install scipy
RUN pip3 install grpcio==1.21.1
RUN pip3 install pyyaml>=3.13
RUN pip3 install pandas>=0.23.1
RUN pip3 install s3fs>=0.2.0
RUN pip3 install segmentation-models>=1.0.0b1
RUN pip3 install pycuda

ENTRYPOINT ["/bin/bash"]