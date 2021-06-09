#!/bin/bash

readonly CORPUS=640k.txt
readonly LOG=out/log.txt

rm -f "$LOG"
make clean
make	ARG_ACTIONS="TRAIN,SAVE,TEST" \
		ARG_TRAIN="res/train/$CORPUS" \
		ARG_TEST="res/test/$CORPUS" \
		ARG_STOP="res/nltk_stop_words.txt" >> "$LOG"
grep Loss "$LOG"
