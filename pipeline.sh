#!/bin/bash

# Declare path constants
readonly DDIR=./data
readonly ODIR=./out
readonly QA=$DDIR/xz/small_segmented_questions_and_answers
readonly Q=$DDIR/questions
readonly A=$DDIR/answers
readonly S=$DDIR/nltk_stop_words.txt
readonly V=$ODIR/vocab.tsv
readonly W=$ODIR/weights-ih.tsv

echo "[PIPELINE] Decmopress questions and answers"
xz -dkf -T0 $QA.xz

echo "[PIPELINE] Format questions and answers"
tr '[:upper:]' '[:lower:]' < $QA > $QA.new && mv $QA.new $QA
perl -pi -e 's/[^\w\s|\b=~=~>\b]//g' $QA
for c in {a..z}; do perl -pi -e 's/'$c'{3,}/'$c$c'/g' $QA; done

echo "[PIPELINE] Separate questions and answers"
grep -E -o '^.* =~=~>' $QA > $Q
perl -pi -e 's/ =~=~>//g' $Q
grep -E -o '=~=~>.*$' $QA > $A
perl -pi -e 's/=~=~> //g' $A
rm $QA

echo "[PIPELINE] Filter questions"
python3 filter.py $Q $A $Q.fil $A.fil 3 5
mv $Q.fil $Q
mv $A.fil $A

echo "[PIPELINE] Embed questions"
make clean
make ARG_TRAIN="$Q" ARG_TEST="/dev/null" ARG_STOP="$S"

echo "[PIPELINE] Encode questions"
python3 encoder.py $Q $A $QA.enc $V $W
rm $Q $A

echo "[PIPELINE] Compress encoded questions and answers"
xz -zf -T0 -0 $QA.enc
