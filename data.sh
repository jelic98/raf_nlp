#!/bin/bash

# Declare path constants
readonly DIR=.
readonly DDIR=./data
readonly D=$DIR/data.zip
readonly S=$DDIR/nltk_stop_words.txt
readonly URL=https://drive.google.com/uc?id=1EP6A8k4pOssRP4bNKmrrT3dGQnPBhfdf

echo "[DATA] Download dataset and stop words"
pip3 install gdown
pip3 install nltk
python3 data.py $URL $D $S
unzip $D && rm $D
sed 's/[^a-z]//g' $S > $S.sed
sort $S.sed > $S
rm $S.sed
rm .DS_Store
rm -rf __MACOSX/
