FROM nvcr.io/nvidia/l4t-base:r32.3.1
 
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

RUN pip3 install keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 enum34 futures protobuf
RUN python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.14.0
CMD [ "bash" ]