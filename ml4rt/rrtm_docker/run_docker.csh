#!/bin/csh

# This script makes mounting/running the RT_docker container "easy"

if($#argv != 1) then
  echo "USAGE: $argv[0] image_id"
  echo "          where image_id : is the identification number or name of the Docker image to execute"
  echo " Note: the current directory will be 'mounted' as /home/user/data"
  exit
endif

echo "Running docker container in interactive mode"

docker run -it --userns=host -u `id -u`:`id -g` -v ${PWD}:/home/user/data ${argv[1]} 
