#ifndef H_NN_INCLUDE
#define H_NN_INCLUDE

#include "lib.h"

// PRIMARY HYPERPARAMETERS
#define THREAD_MAX 2
#define HIDDEN_MAX 50
#define EPOCH_MAX 10
#define WINDOW_MAX 5
#define SAMPLE_MAX 25
#define LEARNING_RATE_MAX 0.05
#define LEARNING_RATE_MIN 0.0001
#define FILTER_LOW_BOUND 5
#define FILTER_HIGH_RATIO 0.05

// SECONDARY HYPERPARAMETERS
#define LOG_PERIOD_PASS 1000
#define LOG_PERIOD_CORPUS 500
#define LOG_PERIOD_SENT 100
#define VOCABULARY_HASH_MAX 30000000
#define PREDICTION_MAX 5
#define FREQ_BUCKETS 10
#define INVALID_INDEX_MAX 10
#define BACKTRACE_DEPTH 10

// Special characters
#define SENTENCE_DELIMITERS ".?!\n\r"
#define WORD_DELIMITERS " \t,:;(){}[]<>\"'’/\\%#$&~*+=^_"

// Number of characters in path
#define PATH_CHARACTER_MAX 255

// Number of characters in line
#define LINE_CHARACTER_MAX 512

// Number of sentences to allocate per file initially
#define SENTENCE_THRESHOLD 128

// Flags
#define FLAG_FILTER_VOCABULARY_STOP
#define FLAG_FILTER_VOCABULARY_LOW
#define FLAG_FILTER_VOCABULARY_HIGH
#define FLAG_UNIGRAM_SAMPLING
#define FLAG_TEST_SIMILARITY
//#define FLAG_TEST_ORTHANT
#define FLAG_TEST_CONTEXT
//#define FLAG_BINARY_INPUT
//#define FLAG_BINARY_OUTPUT
//#define FLAG_SENT
//#define FLAG_STEM
//#define FLAG_FREE_MEMORY

// Paths
#define TRAIN_PATH arg_train
#define TEST_PATH arg_test
#define STOP_PATH arg_stop
#define VOCABULARY_PATH "out/vocab.tsv"
#define WEIGHTS_IH_PATH "out/weights-ih.tsv"
#define WEIGHTS_HO_PATH "out/weights-ho.tsv"
#define SENT_PATH "out/sent.tsv"
#define FILTER_PATH "out/filter.tsv"

// Messages
#define ERROR_FILE "File error occurred"
#define ERROR_MEMORY "Memory error occurred"
#define ERROR_COMMAND "Command error occurred"
#define ERROR_CMDARGS "Command line arguments error occurred"

// Shortcuts
#define DEF_LINE(x) static dt_int(x) = __LINE__
#define ARGS_COUNT (__ARGS_END__ - __ARGS_START__)

// Internal data limits
#define DT_FLOAT_MIN DBL_MIN

// Internal data types
typedef char dt_char;
typedef int dt_int;
typedef unsigned int dt_uint;
typedef unsigned long long dt_ull;
typedef double dt_float;

typedef struct xWord {
	dt_char* word;
	dt_int index;
	dt_ull context_max;
	dt_ull freq;
	dt_float prob;
	dt_float dist;
	dt_ull* target_freq;
	struct xWord* left;
	struct xWord* right;
	struct xWord* next;
	struct xWord** target;
	struct xContext* context;
} xWord;

typedef struct xContext {
	xWord* word;
	dt_ull freq;
	struct xContext* left;
	struct xContext* right;
} xContext;

typedef struct xSent {
	dt_char* sent;
	dt_float* vec;
	dt_float dist;
} xSent;

typedef struct xThread {
	dt_uint id;
	dt_int p;
	xWord* center;
} xThread;

// API functions
void nn_start();
void nn_finish();
void training_run();
void testing_run();
void weights_save();
void weights_load();

// Command line arguments
DEF_LINE(__ARGS_START__);
extern dt_char arg_actions[PATH_CHARACTER_MAX];
extern dt_char arg_train[PATH_CHARACTER_MAX];
extern dt_char arg_test[PATH_CHARACTER_MAX];
extern dt_char arg_stop[PATH_CHARACTER_MAX];
DEF_LINE(__ARGS_END__);

// Global variables
dt_int pattern_max, input_max, hidden_max, output_max;

// Dependencies
#include "log.h"
#include "mat.h"
#include "col.h"
#include "voc.h"

#ifdef FLAG_SENT
#include "sent.h"
#endif

#ifdef FLAG_STEM
#include "stmr.h"
#endif
#endif
