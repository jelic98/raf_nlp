# Declare path constants
readonly QA=out/segmented_questions_and_answers
readonly Q=out/questions
readonly A=out/answers
readonly V=out/vocab.tsv
readonly W=out/weights-ih.tsv

# Decmopress questions and answers
xz -dkf -T0 $QA.xz

# Formatting
tr '[:upper:]' '[:lower:]' < $QA > $QA.new && mv $QA.new $QA
perl -pi -e 's/[^\w\s|\b=~=~>\b]//g' $QA
for c in {a..z}; do perl -pi -e 's/'$c'{3,}/'$c$c'/g' $QA; done

# Separate questions and answers
grep -E -o '^.* =~=~>' $QA > $Q
perl -pi -e 's/ =~=~>//g' $Q
grep -E -o '=~=~>.*$' $QA > $A
perl -pi -e 's/=~=~> //g' $A
rm $QA

# Remove stop words from questions and filter by frequency
python3 filter.py $Q $A $Q.fil $A.fil 3 5
mv $Q.fil $Q
mv $A.fil $A

# Compile and run embedder
make

# Perform sentence encoding
python3 encoder.py $Q $A $QA $V $W

# Compress questions, vectors and answers
xz -zkf -T0 -0 $QA
