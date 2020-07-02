## TODO

Features to be implemented in no particular order.

### Model

* raise frequencies to the power of 3/4 before flattening vocabulary
* initialize input weights inverse proportionally to word frequency
* check weights normalization

### Parser

* [URGENT] alphabetical reorder changes word indices (compare indices instead of word in testing)

### Other

* add pmi to track context connections to avaid bst context contains
* scan vocabulary if word is not found in the map
* sort vocabulary by frequency for faster retrievals
* iterative functions for bst
* multithreading?

### Random

* [word2vec in C](https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c)
* [word2vec in Python](https://github.com/deborausujono/word2vecpy/blob/master/word2vec.py)
* [word2vec critic](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec)
* [Embedding Projector](https://projector.tensorflow.org)
