FROM 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-dev

ARG CUDNN_VERSION=7.5.0.56
ENV ENV_CUDNN_VERSION=$CUDNN_VERSION
#ENV CUDNN_VERSION 7.5.0.56
ENV CUDNN_PKG_VERSION=${CUDNN_VERSION}-1
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

COPY libcudnn7_$CUDNN_VERSION-1+cuda10.0_arm64.deb .
COPY libcudnn7-dev_$CUDNN_VERSION-1+cuda10.0_arm64.deb .
COPY libcudnn7-doc_$CUDNN_VERSION-1+cuda10.0_arm64.deb .

RUN dpkg -i libcudnn7_${CUDNN_VERSION}-1+cuda10.0_arm64.deb && \
    dpkg -i libcudnn7-dev_${CUDNN_VERSION}-1+cuda10.0_arm64.deb && \
    dpkg -i libcudnn7-doc_$CUDNN_VERSION-1+cuda10.0_arm64.deb && \
    rm libcudnn7_$CUDNN_VERSION-1+cuda10.0_arm64.deb && \
    rm libcudnn7-dev_$CUDNN_VERSION-1+cuda10.0_arm64.deb && \
    rm libcudnn7-doc_$CUDNN_VERSION-1+cuda10.0_arm64.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["/bin/bash"]