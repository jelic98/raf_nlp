#!/bin/bash

# Declare path constants
readonly DIR=.
readonly DDIR=./data
readonly D=$DIR/data.zip
readonly S=$DDIR/nltk_stop_words.txt
readonly URL=https://drive.google.com/uc?id=1EP6A8k4pOssRP4bNKmrrT3dGQnPBhfdf

echo "[DATA] Download dataset"
pip3 install gdown
python3 data.py $URL $D
unzip $D && rm $D && rm -rf __MACOSX/

echo "[DATA] Download NLTK stop words"
pip3 install nltk
python3 stop.py $S
sed 's/[^a-z]//g' $S > $S.sed
sort $S.sed > $S
rm $S.sed
