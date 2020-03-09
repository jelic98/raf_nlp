#ifndef H_NN_INCLUDE
#define H_NN_INCLUDE

#include "include/main.h"

#define EPOCH_MAX 10
#define HIDDEN_MAX 50
#define WINDOW_MAX 10
#define INITIAL_WEIGHT_MAX 0.5
#define LEARNING_RATE_MAX 0.025
#define LEARNING_RATE_MIN 0.001
#define INVALID_INDEX_MAX 10

#define SENTENCE_DELIMITERS ".?!"
#define WORD_DELIMITERS " \t\n\r,:;(){}[]<>\"'’/\\%#$&~*+=^_"

// Number of sentences in file
#define SENTENCE_MAX 300

// Number of words in sentence
#define WORD_MAX 100

// Number of characters in word
#define CHARACTER_MAX 50

// Number of characters in line
#define LINE_CHARACTER_MAX 512

// Flags
#define FLAG_DEBUG
//#define FLAG_LOG
//#define FLAG_PRINT_VOCAB
//#define FLAG_PRINT_ERRORS

// Paths
#define CORPUS_PATH "res/corpus-large.txt"
#define FILTER_PATH "res/filter.txt"
#define WEIGHTS_IH_PATH "out/weights-ih.txt"
#define WEIGHTS_HO_PATH "out/weights-ho.txt"
#define LOG_PATH "out/log.txt"

// Messages
#define FILE_ERROR_MESSAGE "File error occurred\n"

// Shortcuts
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define random() ((double)rand() / ((double) RAND_MAX + 1))

// Internal data types
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

// API functions
void nn_start();
void nn_finish();
void training_run();
void test_run(char*, int, int*);
void weights_save();
void weights_load();
void sentence_encode(char*, double*);

// Test cases
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
#endif
