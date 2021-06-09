#!/bin/bash

readonly CORPUS=640k.txt
readonly LOG=out/log.txt

rm -f "$LOG"
make clean
make	ARG_ACTIONS="TRAIN,SAVE,TEST" \
		ARG_TRAIN="data/train/$CORPUS" \
		ARG_TEST="data/test/$CORPUS" \
		ARG_STOP="data/nltk_stop_words.txt" >> "$LOG"
grep Loss "$LOG"
