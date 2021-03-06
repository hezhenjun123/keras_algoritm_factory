#!/bin/bash

uname -m
set -x
xhost +

login=$(aws ecr get-login --no-include-email)
$login

docker image prune -f

docker_name="zoomlion:local-dev"
aws_creds="${HOME}/.aws/credentials"
aws_env=""

if [ -f $aws_creds  ]
then
	aws_id="$(cat $aws_creds | grep id | cut -f2 -d= | sed 's# ##g')"
	aws_secret="$(cat $aws_creds | grep secret | cut -f2 -d= | sed 's# ##g')"
	aws_env="-e AWS_ACCESS_KEY_ID=${aws_id} -e AWS_SECRET_ACCESS_KEY=${aws_secret}"
else
	echo "Please run `aws configure`, ~/.aws/credentials are required for this to run"
	exit 1
fi

hostname_prefix=$(hostname | cut -f1 -d-)
if [ $hostname_prefix == "ip" ]
then
	# if on an ec2 machine, do not forward anything, just let it run
	echo "Detected an EC2 instance, no running display or webcam forwarding to containers"
	display_str=""
	webcam_str=""
else
	# forward display into docker container
	display_str="--rm -ti --net=host --ipc=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --env=\"DISPLAY\""

	# forward webcam into docker container
	webcam_str="--device=/dev/video0:/dev/video0"
fi

# change flag to F
# docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

case $( uname -m ) in
    x86*)
	nvidia-docker build -t ${docker_name} -f docker-x86/app/Dockerfile .
	;;
    aar*)
	nvidia-docker build -t ${docker_name} -f docker-jetson/app/Dockerfile .
	;;
    *)
	echo -n "unknown platform"
esac


case $( uname -m ) in
    x86*)
	nvidia-docker run ${display_str} ${webcam_str} ${aws_env} ${docker_name} $*
	;;
    aar*)
	nvidia-docker run --entrypoint="" --privileged ${display_str} ${webcam_str} ${aws_env} -it ${docker_name} $*
	;;
    *)
	echo -n "unknown platform"
esac
