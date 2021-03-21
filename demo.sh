#!/bin/bash

make clean
make	ARG_ACTIONS="LOAD,TRAIN,SAVE" \
		ARG_TRAIN="data/train/bill_gates.txt" \
		ARG_TEST="data/test/bill_gates.txt" \
		ARG_STOP="data/nltk_stop_words.txt"
