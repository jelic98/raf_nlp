[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Usage

1. Grant execution privilege to helper scripts
```bash
chmod +x data.sh pipeline.sh
```

2. Download dataset
```bash
./data.sh
```

3. Configure hyperparameters
```bash
vim include/nn.h
```

4a. Start training
```bash
./run.sh
```

4b. Start training inside pipeline
```bash
./pipeline.sh
```

## Links

* [word2vec in C](https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c)
* [word2vec in Python](https://github.com/deborausujono/word2vecpy/blob/master/word2vec.py)
* [word2vec critic](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec)
* [Embedding Projector](https://projector.tensorflow.org)
