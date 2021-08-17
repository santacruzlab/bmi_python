#!/bin/bash
SOURCE_PATH=`realpath ..`
WORK_VOLUME=/storage/
DOCKER_IMG=bmi3d:latest

# for graphics 	--device /dev/arduino_neurosync \
export DISPLAY=:0.0
xhost +local:docker

docker volume create $WORK_VOLUME     # this will be persistent every time the image is invoked

docker run --rm -ti \
	--device /dev/snd \
	--device /dev/dri:/dev/dri \
	--device /dev/arduino_joystick \
	--device /dev/arduino_neurosync \
    -v $WORK_VOLUME:/storage \
    -v $SOURCE_PATH:/src \
    -w /work \
    -e DISPLAY=:0.0 -v /tmp/.X11-unix:/tmp/.X11-unix \
    -p 8000:8000 \
    $DOCKER_IMG bash 
    
