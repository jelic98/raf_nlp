#!/bin/bash

readonly LOG=out/log.txt

nohup sh build.sh 2>&1 > "$LOG" &
watch -n 1 --color "cat $LOG | tail -n $(($(tput lines) - 2))"

grep "Loss" "$LOG"
grep "Finished training" "$LOG"
