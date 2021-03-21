#!/bin/bash

make clean
make	ARG_ACTIONS="LOAD,TRAIN,SAVE" \
		ARG_TRAIN="data/train/text8.txt" \
		ARG_TEST="data/test/text8.txt" \
		ARG_STOP="data/nltk_stop_words.txt"
