FROM 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-dev

ARG TRT_VERSION=5.1.6
ARG TRT_VERSION_EXT=${TRT_VERSION}.1
LABEL com.nvidia.trt.version="${TRT_VERSION}"

COPY graphsurgeon-tf_$TRT_VERSION-1+cuda10.0_arm64.deb .
COPY tensorrt_$TRT_VERSION_EXT-1+cuda10.0_arm64.deb .
COPY uff-converter-tf_$TRT_VERSION-1+cuda10.0_arm64.deb .

RUN dpkg -i graphsurgeon-tf_$TRT_VERSION-1+cuda10.0_arm64.deb && \
    dpkg -i tensorrt_$TRT_VERSION_EXT-1+cuda10.0_arm64.deb && \
    dpkg -i uff-converter-tf_$TRT_VERSION-1+cuda10.0_arm64.deb && \
    rm graphsurgeon-tf_$TRT_VERSION-1+cuda10.0_arm64.deb && \
    rm tensorrt_$TRT_VERSION_EXT-1+cuda10.0_arm64.deb && \
    rm uff-converter-tf_$TRT_VERSION-1+cuda10.0_arm64.deb

ENTRYPOINT ["/bin/bash"]