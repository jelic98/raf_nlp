#ifndef H_NN_INCLUDE
#define H_NN_INCLUDE

#include "include/main.h"

#define EPOCH_MAX 1
#define HIDDEN_MAX 50
#define WINDOW_MAX 10
#define NEGATIVE_SAMPLES_MAX 6
#define INITIAL_WEIGHT_MAX 0.5
#define LEARNING_RATE_MAX 0.025
#define LEARNING_RATE_MIN 0.001

#define MONTE_CARLO_FACTOR 0.1
#define MONTE_CARLO_EMERGENCY 10
#define INVALID_INDEX_MAX 10

#define SENTENCE_DELIMITERS ".?!"
#define WORD_DELIMITERS " \t\n\r,:;(){}[]<>\"'’/\\%#$&~*+=^_"

// Number of characters in line
#define LINE_CHARACTER_MAX 512

// Number of sentences to allocate per file initially
#define SENTENCE_THRESHOLD 64

// Number of words to allocate per sentence initially
#define WORD_THRESHOLD 16

// Flags
//#define FLAG_NEGATIVE_SAMPLING
#define FLAG_DEBUG
#define FLAG_LOG
#define FLAG_PRINT_CORPUS
//#define FLAG_PRINT_ERRORS

// Paths
#define CORPUS_PATH "res/corpus/large.txt"
#define TRAIN_PATH "res/train/large.txt"
#define TEST_PATH "res/test/large.txt"
#define STOP_PATH "res/misc/stop.txt"
#define WEIGHTS_IH_PATH "out/weights-ih.txt"
#define WEIGHTS_HO_PATH "out/weights-ho.txt"
#define LOG_PATH "out/log.txt"

// Messages
#define FILE_ERROR_MESSAGE "File error occurred\n"

// Shortcuts
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define random(a, b) ((rand() / (dt_float) RAND_MAX) * (b - a) + a)

// Internal data limits
#define DT_FLOAT_MIN DBL_MIN

// Internal data types
typedef char dt_char;
typedef int dt_int;
typedef unsigned int dt_uint;
typedef double dt_float;

typedef struct xWord {
	dt_char* word;
	dt_uint freq;
	dt_uint context_count;
	dt_float prob;
	struct xWord* left;
	struct xWord* right;
} xWord;

typedef union xBit {
	dt_uint on : 1;
} xBit;

// API functions
void nn_start();
void nn_finish();
void training_run();
void test_run();
void weights_save();
void weights_load();
void sentence_encode(dt_char*, dt_float*);
#endif
