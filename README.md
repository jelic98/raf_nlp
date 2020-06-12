## TODO

Features to be implemented in no particular order.

### Model

* loss in negative sampling is increasing over time
* check weights normalization
* initialize input weights inverse proportionally to word frequency
* adam optimizer?
* mini batching?

### Parser

* [URGENT] alphabetical reorder changes word indices (compare indices instead of word in testing)

### Other

* add pmi to track context connections to avaid bst context contains
* scan vocabulary if word is not found in the map
* sort vocabulary by frequency for faster retrievals
* iterative functions for bst
* multithreading

### Random

* [word2vec in C](https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c)
* [word2vec in Python](https://github.com/deborausujono/word2vecpy/blob/master/word2vec.py)
* [word2vec critic](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec)
* [Embedding Projector](https://projector.tensorflow.org)
