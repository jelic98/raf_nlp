#!/bin/bash

make clean
make	ARG_ACTIONS="TRAIN,SAVE" \
		ARG_TRAIN="data/train/gates.txt" \
		ARG_TEST="data/test/gates.txt" \
		ARG_STOP="data/nltk_stop_words.txt"
