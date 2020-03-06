#ifndef H_NN_INCLUDE
#define H_NN_INCLUDE

#include "include/main.h"

#define HIDDEN_MAX 100
#define WINDOW_MAX 20

#define LEARNING_RATE_MAX 0.025
#define LEARNING_RATE_MIN 0.001

#define INITIAL_WEIGHT_MAX 0.5

#define EPOCH_MAX 1

#define LOG_EPOCH 1
#define LOG_PERIOD 1

#ifdef H_TEST_INCLUDE
const int TEST_CASES_START = __LINE__;
#define TEST_CASES\
	"gates",\
	"president",\
	"computer",\
	"seattle",\
	"software",\
	"ceo",\
	"foundation",\
	"ibm",\
	"apple",\
	"microsoft"
const int TEST_CASES_END = __LINE__;
#define TEST_MAX (TEST_CASES_END - TEST_CASES_START - 2)
#endif

#define CORPUS_PATH "res/corpus-large.txt"
#define FILTER_PATH "res/filter.txt"
#define WEIGHTS_IH_PATH "out/weights-ih.txt"
#define WEIGHTS_HO_PATH "out/weights-ho.txt"
#define LOG_PATH "out/log.txt"

#define FILE_ERROR_MESSAGE "File error occurred\n"

// Number of sentences in file
#define SENTENCE_MAX 1000

// Number of words in sentence
#define WORD_MAX 50

// Number of characters in word
#define CHARACTER_MAX 25

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
void get_predictions(const char*, unsigned int, unsigned int*);
void load_weights();
void save_weights();

#endif
