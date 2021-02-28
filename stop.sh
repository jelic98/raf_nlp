#!/bin/bash

# Declare path constants
readonly DIR=./data
readonly S=$DIR/nltk_stop_words.txt

echo "[STOP] Download NLTK stop words"
python3 stop.py $S
sed 's/[^a-z]//g' $S > $S.sed
sort $S.sed > $S
rm $S.sed
