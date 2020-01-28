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

nvidia-docker build -t ${docker_name} -f docker-x86/app/Dockerfile .

if [ $hostname_prefix == "ip" ]
then
   nvidia-docker run ${display_str} ${webcam_str} ${aws_env} -v ${HOME}/sandbox/landing-shared-workspace:/root/sandbox/landing-shared-workspace ${docker_name} $*
else
   nvidia-docker run ${display_str} ${webcam_str} ${aws_env} -it ${docker_name} $*
fi



# docker tag local/jetson:v2 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:v0.0.1
# login=$(aws ecr get-login --no-include-email)
# docker tag local/jetson:v2 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:v0.0.1
