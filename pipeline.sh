#!/bin/bash
source import.sh

# Declare path constants
readonly IN=small_segmented_questions_and_answers
readonly DDIR=./data
readonly ODIR=./out
readonly QA=$DDIR/xz/$IN
readonly Q=$DDIR/questions
readonly A=$DDIR/answers
readonly S=$DDIR/nltk_stop_words.txt
readonly V=$ODIR/vocab.tsv
readonly W=$ODIR/weights-ih.tsv

log "Clean workspace"
rm -f $DDIR/enc/$IN.zip $QA.enc.xz $QA.enc $QA.low $QA $Q.fil $Q $A $V $W

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

log "Embed questions"
make clean
make ARG_TRAIN="$Q" ARG_STOP="$S"

log "Encode questions"
python3 encoder.py $Q $A $QA.enc $V $W
rm -f $Q $A

log "Compress encoded questions and answers"
xz -zf -T0 -0 $QA.enc

log "Export data for indexing"
mkdir -p $DDIR/enc
mkdir -p $IN
cp $QA.enc.xz $IN/corpus.enc.xz
cp $V $IN/vocab.tsv
cp $W $IN/embed.tsv
zip -q $DDIR/enc/$IN.zip $IN/corpus.enc.xz $IN/vocab.tsv $IN/embed.tsv
rm -rf $IN
