#!/bin/bash

# Declare path constants
readonly DATE="date +'%d-%m-%Y %H:%M:%S'"
readonly DDIR=./data
readonly ODIR=./out
readonly QA=$DDIR/xz/segmented_questions_and_answers
readonly Q=$DDIR/questions
readonly A=$DDIR/answers
readonly S=$DDIR/nltk_stop_words.txt
readonly V=$ODIR/vocab.tsv
readonly W=$ODIR/weights-ih.tsv

printf "\033[0;36m"

echo "[$(eval $DATE)] [PIPELINE] Clean workspace"
rm -f $QA.enc.xz $QA.enc $QA.low $QA $Q.fil $Q $A $V $W

echo "[$(eval $DATE)] [PIPELINE] Decmopress questions and answers"
xz -dkf -T0 $QA.xz

echo "[$(eval $DATE)] [PIPELINE] Format questions and answers"
tr '[:upper:]' '[:lower:]' < $QA > $QA.low && mv $QA.low $QA
perl -pi -e 's/[^\w\s|\b=~=~>\b]//g' $QA
for c in {a..z}; do perl -pi -e 's/'$c'{3,}/'$c$c'/g' $QA; done

echo "[$(eval $DATE)] [PIPELINE] Separate questions and answers"
grep -E -o '^.* =~=~>' $QA > $Q
perl -pi -e 's/ =~=~>//g' $Q
grep -E -o '=~=~>.*$' $QA > $A
perl -pi -e 's/=~=~> //g' $A
rm -f $QA

echo "[$(eval $DATE)] [PIPELINE] Filter questions"
python3 filter.py $Q $Q.fil 3 0.001
mv $Q.fil $Q

echo "[$(eval $DATE)] [PIPELINE] Embed questions"
make clean
make ARG_TRAIN="$Q" ARG_TEST="/dev/null" ARG_STOP="$S"

echo "[$(eval $DATE)] [PIPELINE] Encode questions"
python3 encoder.py $Q $A $QA.enc $V $W
rm -f $Q $A

echo "[$(eval $DATE)] [PIPELINE] Compress encoded questions and answers"
xz -zf -T0 -0 $QA.enc

printf "\033[0m"
