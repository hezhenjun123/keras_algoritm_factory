FROM 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-dev

WORKDIR /app

ARG INF_VERSION=5.1.6
ENV ENV_INF_VERSION=$INF_VERSION
ENV INF_PKG_VERSION=${INF_VERSION}-1
LABEL com.nvidia.inf.version="${INF_VERSION}"

COPY libnvinfer-dev_$INF_VERSION-1+cuda10.0_arm64.deb .
COPY libnvinfer5_$INF_VERSION-1+cuda10.0_arm64.deb .
COPY libnvinfer-samples_$INF_VERSION-1+cuda10.0_all.deb  .
COPY python3-libnvinfer_$INF_VERSION-1+cuda10.0_arm64.deb .
COPY python3-libnvinfer-dev_$INF_VERSION-1+cuda10.0_arm64.deb .

RUN dpkg -i libnvinfer5_$INF_VERSION-1+cuda10.0_arm64.deb && \
    dpkg -i libnvinfer-dev_$INF_VERSION-1+cuda10.0_arm64.deb && \
    dpkg -i libnvinfer-samples_$INF_VERSION-1+cuda10.0_all.deb && \
    dpkg -i python3-libnvinfer_$INF_VERSION-1+cuda10.0_arm64.deb && \
    dpkg -i python3-libnvinfer-dev_$INF_VERSION-1+cuda10.0_arm64.deb && \
    rm libnvinfer5_$INF_VERSION-1+cuda10.0_arm64.deb && \
    rm libnvinfer-dev_$INF_VERSION-1+cuda10.0_arm64.deb && \
    rm libnvinfer-samples_$INF_VERSION-1+cuda10.0_all.deb && \
    rm python3-libnvinfer_$INF_VERSION-1+cuda10.0_arm64.deb && \
    rm python3-libnvinfer-dev_$INF_VERSION-1+cuda10.0_arm64.deb

ENTRYPOINT ["/bin/bash"]