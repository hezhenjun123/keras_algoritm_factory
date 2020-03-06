FROM 286751717145.dkr.ecr.us-east-2.amazonaws.com/zoomlion:x86-tensorrt-5.1.5-py36
# packages not supported on miivii yet
RUN python3 -m pip install sklearn 
COPY . /app
WORKDIR /app
