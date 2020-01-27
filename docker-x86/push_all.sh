#!/bin/bash

login=$(aws ecr get-login --no-include-email)
$login

#https://docs.nvidia.com/deeplearning/sdk/tensorrt-container-release-notes/rel_19-06.html#rel_19-08
cd tensorrt
docker build -t 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomline:x86-tensorrt-5.1.5-py36 .
docker push 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomline:x86-tensorrt-5.1.5-py36
cd ..


