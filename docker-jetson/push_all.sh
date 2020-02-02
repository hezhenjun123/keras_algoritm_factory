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

cp ./debs/libnvinfer* ./nvinfer/
cp ./debs/python3-libnvinfer* ./nvinfer/
cp ./debs/tensorrt* ./tensorrt/
cp ./debs/graphsurgeon* ./tensorrt/
cp ./debs/uff-converter* ./tensorrt/
cp ./debs/libcudnn* ./cudnn/
cp ./debs/cuda*.zip ./opencv/
#mv ./debs/cuda*.patch ./opencv/
cp ./debs/cuda* ./cuda/

cd cuda
#podman build -v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-dev .
podman build -v $(pwd)/../qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-dev .
podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-dev
cd ..

cd cudnn
#podman build -v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-dev .
ls -lth 
podman build -v $(pwd)/../qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-dev .
podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-dev
cd ..


cd nvinfer
podman build -v $(pwd)/../qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-dev .
podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-dev
cd ..

cd tensorrt
podman build -v $(pwd)/../qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-dev .
podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-dev
cd ..

cd opencv
podman build -v $(pwd)/../qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-opencvdev .
podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-dev
cd ..

cd tensorflow
podman build -v $(pwd)/../qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-tensorflow-dev .
podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-tensorflow-dev
cd ..


