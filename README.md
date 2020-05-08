## TODO

Features to be implemented in no particular order.

### Model

* normally distribute indexes by 1/freq to form unigram table
* sort vocabulary by frequency for faster fetrievals
* cross validation
* adam optimizer?
* flag for fixed weights initialization
* normalize word vectors
* normalize sentence vectors

### Other

* scan through vocabulary if word is not found in the map
* file read/write binary data instad of text
* match every malloc with free (bug in 3.txt corpus)
* multithreading
* all api functions should be void
* isolate modules into separate headers

### Note
* vocabulary reduction is not required

### Random

* [word2vec in C](https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c)
* [word2vec in Python](https://github.com/deborausujono/word2vecpy/blob/master/word2vec.py)
