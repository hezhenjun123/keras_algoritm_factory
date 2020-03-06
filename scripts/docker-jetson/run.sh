# change flag to F
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

docker tag local/jetson:v2 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:v0.0.1
login=$(aws ecr get-login --no-include-email)
docker tag local/jetson:v2 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:v0.0.1
