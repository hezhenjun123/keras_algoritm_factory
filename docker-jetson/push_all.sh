#!/bin/bash

cd tensorflow
podman build -v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomline:jetson-tensorflow-1.14.0 .
podman push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomline:jetson-tensorflow-1.14.0
cd ..


