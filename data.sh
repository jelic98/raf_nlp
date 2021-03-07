#!/bin/bash

# Declare path constants
readonly DATE="date +'%d-%m-%Y %H:%M:%S'"
readonly DIR=.
readonly DDIR=./data
readonly D=$DIR/data.zip
readonly S=$DDIR/nltk_stop_words.txt
readonly URL=https://drive.google.com/uc?id=1EP6A8k4pOssRP4bNKmrrT3dGQnPBhfdf

echo "[$(eval $DATE)] [PIPELINE] Clean workspace"
rm -f $S

echo "[$(eval $DATE)] [DATA] Download libraries"
python3 -m pip install --upgrade pip
pip3 install gdown
pip3 install nltk
pip3 install numpy

echo "[$(eval $DATE)] [DATA] Download dataset and stop words"
python3 data.py $URL $D $S
unzip $D && rm $D
sed 's/[^a-z]//g' $S > $S.sed
sort $S.sed > $S
rm -rf __MACOSX/ .DS_Store $S.sed
