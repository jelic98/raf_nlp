#!/bin/bash

# Declare path constants
readonly DIR=.
readonly D=$DIR/data.zip
readonly URL=https://drive.google.com/uc?id=1fGYzEkGSTLBDFSRRiQw9jeaC9xHWws2g

echo "[DATA] Download dataset"
pip install gdown
python3 data.py $D
unzip $D && rm $D
