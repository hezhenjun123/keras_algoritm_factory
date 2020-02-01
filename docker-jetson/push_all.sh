#!/bin/bash

login=$(aws ecr get-login --no-include-email)
$login

cd cuda
podman build -v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda .
podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda
cd ..

cd cudnn
podman build -v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn .
podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-xavier-l4t-cuda-cudnn
cd ..

cd tensorflow
podman build -v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-tensorflow-1.14.0 .
podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:jetson-tensorflow-1.14.0
cd ..


