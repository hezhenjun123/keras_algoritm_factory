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

#mv ./debs/libnvinfer* ./nvinfer/
#mv ./debs/python3-libnvinfer* ./nvinfer/
#mv ./debs/tensorrt* ./tensorrt/
#mv ./debs/graphsurgeon* ./tensorrt/
#mv ./debs/uff-converter* ./tensorrt/
mv ./debs/libcudnn* ./cudnn/
#mv ./debs/cuda*.zip ./opencv/
#mv ./debs/cuda*.patch ./opencv/
mv ./debs/cuda* ./cuda/

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

# cd tensorflow
# podman build -v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-tensorflow-1.14.0 .
# podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-tensorflow-1.14.0
# cd ..


