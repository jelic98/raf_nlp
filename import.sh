#!/bin/bash

readonly DATE="date +'%d-%m-%Y %H:%M:%S'"

log() {
	printf "\033[0;36m"
	file=$(basename $0)
	echo "[$(eval $DATE)] [$(echo "${file%.*}" | tr a-z A-Z)] $1"
	printf "\033[0m"
}
