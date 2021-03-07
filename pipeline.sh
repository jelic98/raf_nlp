#!/bin/bash
source import.sh

# Declare path constants
readonly DDIR=./data
readonly ODIR=./out
readonly QA=$DDIR/xz/segmented_questions_and_answers
readonly Q=$DDIR/questions
readonly A=$DDIR/answers
readonly S=$DDIR/nltk_stop_words.txt
readonly V=$ODIR/vocab.tsv
readonly W=$ODIR/weights-ih.tsv

log "Clean workspace"
rm -f $QA.enc.xz $QA.enc $QA.low $QA $Q.fil $Q $A $V $W

log "Decmopress questions and answers"
xz -dkf -T0 $QA.xz

log "Format questions and answers"
tr '[:upper:]' '[:lower:]' < $QA > $QA.low && mv $QA.low $QA
perl -pi -e 's/[^\w\s|\b=~=~>\b]//g' $QA
for c in {a..z}; do perl -pi -e 's/'$c'{3,}/'$c$c'/g' $QA; done

log "Separate questions and answers"
grep -E -o '^.* =~=~>' $QA > $Q
perl -pi -e 's/ =~=~>//g' $Q
grep -E -o '=~=~>.*$' $QA > $A
perl -pi -e 's/=~=~> //g' $A
rm -f $QA

log "Filter questions"
python3 filter.py $Q $Q.fil 3 0.001
mv $Q.fil $Q

log "Embed questions"
make clean
make ARG_TRAIN="$Q" ARG_TEST="/dev/null" ARG_STOP="$S"

log "Encode questions"
python3 encoder.py $Q $A $QA.enc $V $W
rm -f $Q $A

log "Compress encoded questions and answers"
xz -zf -T0 -0 $QA.enc
