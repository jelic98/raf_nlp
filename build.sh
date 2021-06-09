#!/bin/bash

readonly CORPUS=text8.txt

make clean
make ARG_ACTIONS="TRAIN,SAVE,TEST" \
	ARG_TRAIN="res/train/$CORPUS" \
	ARG_TEST="res/test/$CORPUS" \
	ARG_STOP="res/nltk_stop_words.txt"
