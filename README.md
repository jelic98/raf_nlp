## Usage

1. Grant execution privilege to a dataset donwload script
```bash
chmod +x res.sh
```

2. Download dataset
```bash
./res.sh
```
3. Configure hyperparameters
```bash
vim include/nn.h
```

4. Start embedder training
```bash
make
```

## Links

* [word2vec in C](https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c)
* [word2vec in Python](https://github.com/deborausujono/word2vecpy/blob/master/word2vec.py)
* [word2vec critic](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec)
* [Embedding Projector](https://projector.tensorflow.org)
