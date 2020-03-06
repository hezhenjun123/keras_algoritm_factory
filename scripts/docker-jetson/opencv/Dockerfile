FROM 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-dev

WORKDIR /app

ARG OCV_VERSION="4.1.1"
#ENV DEBIAN_FRONTEND noninteractive

WORKDIR /opencv
ADD ./* /opencv/

RUN bash build_opencv.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["/bin/bash"]