#ifndef H_NN_INCLUDE
#define H_NN_INCLUDE

#include "include/main.h"

#define PATTERN_MAX 4
#define INPUT_MAX 2
#define HIDDEN_MAX 2
#define OUTPUT_MAX 1

#define WINDOW_MAX 2

#define LEARNING_RATE 0.5
#define INITIAL_WEIGHT_MAX 0.5

#define EPOCH_MAX 10000
#define LOSS_MAX 0.0002

#define LOG_PERIOD 100

#define LOG_FILE stdout

#define CORPUS_FILE "data/corpus.txt"

#define FILE_ERROR_MESSAGE "File error occurred\n"

// Number of sentences in file
#define SENTENCE_MAX 10

// Number of words in sentence
#define WORD_MAX 100

// Number of characters in word
#define CHARACTER_MAX 50

#define random() ((double)rand() / ((double) RAND_MAX + 1))

typedef struct xWord {
	char word[CHARACTER_MAX];
	unsigned int count;
	struct xWord* left;
	struct xWord* right;
} xWord;

typedef struct xBit {
	unsigned int on : 1;
} xBit;

void start_training();

#endif
