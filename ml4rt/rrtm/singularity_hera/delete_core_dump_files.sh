#!/usr/bin/bash

# Deletes core-dump files from the RRTM.  You can do this while the RRTM is running.
# Argument 1 is the name of the top-level directory with RRTM files.

top_directory_name=$1

while :
do
	rm -v ${top_directory_name}/20*/core.*
	sleep 300
done
