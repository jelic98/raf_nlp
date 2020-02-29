// https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c
// https://nathanrooy.github.io/posts/2018-03-22/word2vec-from-scratch-with-python-and-numpy/

#ifndef H_NN_INCLUDE
#define H_NN_INCLUDE

#include "include/main.h"

#define WINDOW_MAX 2

#define LEARNING_RATE 0.05
#define LEARNING_RATE_MIN 0.001

#define INITIAL_WEIGHT_MAX 0.5

#define EPOCH_MAX 30

#define LOG_EPOCH 1
#define LOG_PERIOD 1

#define CORPUS_PATH "res/corpus-tiny.txt"
#define FILTER_PATH "res/filter.txt"
#define WEIGHTS_IH_PATH "out/weights-ih.txt"
#define WEIGHTS_HO_PATH "out/weights-ho.txt"
#define LOG_PATH "out/log.txt"

#define FILE_ERROR_MESSAGE "File error occurred\n"

// Number of sentences in file
#define SENTENCE_MAX 100

// Number of words in sentence
#define WORD_MAX 100

// Number of characters in word
#define CHARACTER_MAX 50

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define random() ((double)rand() / ((double) RAND_MAX + 1))

typedef struct xWord {
	char word[CHARACTER_MAX];
	unsigned int context_count;
	double prob;
	struct xWord* left;
	struct xWord* right;
} xWord;

typedef union xBit {
	unsigned int on : 1;
} xBit;

void start_training();
void finish_training();
void get_predictions(char*, int);
void load_weights();
void save_weights();

#endif
