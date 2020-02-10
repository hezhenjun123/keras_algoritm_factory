#!/bin/bash

docker run --rm --privileged multiarch/qemu-user-static:register --reset
cat /proc/sys/fs/binfmt_misc/qemu-aarch64
docker create -it --name dummy multiarch/qemu-user-static:x86_64-aarch64 bash
docker cp dummy:/usr/bin/qemu-aarch64-static qemu-aarch64-static
ls qemu-aarch64-static
docker rm -f dummy


login=$(aws ecr get-login --no-include-email)
$login

#docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

aws s3 sync s3://zoomlion-prod-data/xavier/debs ./debs

rsync ./debs/libnvinfer* ./nvinfer/
rsync ./debs/python3-libnvinfer* ./nvinfer/
rsync ./debs/tensorrt* ./tensorrt/
rsync ./debs/graphsurgeon* ./tensorrt/
rsync ./debs/uff-converter* ./tensorrt/
rsync ./debs/libcudnn* ./cudnn/
#rsync ./debs/cuda*.zip ./opencv/
rsync ./debs/cuda* ./cuda/
rsync ./debs/Jetson* ./l4t/

stage=0

function usage
{
    echo "usage: ./build_opencv.sh [[-s sourcedir ] | [-h]]"
    echo "-s | --stage choose which test to run (1, 2, 3...)"
    echo "-h | --help  This message"
}


while [ "$1" != "" ]; do
    case $1 in
        -s | --stage )      shift
			                stage=$1
                            ;;
	    -h | --help )       shift
			                ;;
	    * )                 usage
                            exit 1
    esac
    shift
done

if [ $stage -le 0 ]; then
    cd cuda
    podman build -v $(pwd)/../qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-dev .
    podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-dev
    cd ..
fi

if [ $stage -le 1 ]; then
    cd cudnn
    podman build -v $(pwd)/../qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-dev .
    podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-dev
    cd ..
fi

if [$stage -le 2]; then
    cd nvinfer
    podman build -v $(pwd)/../qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-dev .
    podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-dev
    cd ..
fi

if [$stage -le 3]; then
    cd tensorrt
    podman build -v $(pwd)/../qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-dev .
    podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-dev
    cd ..
fi

if [$stage -le 4]; then
    cd opencv
    podman build -v $(pwd)/../qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-opencvdev .
    podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-dev
    cd ..
fi

if [$stage -le 5]; then
    cd tensorflow
    podman build -v $(pwd)/../qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-tensorflow-dev .
    podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-tensorflow-dev
    cd ..
fi


