#ifndef H_NN_INCLUDE
#define H_NN_INCLUDE

#include "lib.h"

// PRIMARY HYPERPARAMETERS
#define THREAD_MAX 32
#define EPOCH_MAX 20
#define HIDDEN_MAX 200
#define WINDOW_MAX 10
#define NEGATIVE_SAMPLES_MAX 25
#define LEARNING_RATE_FIX 0.05
#define LEARNING_RATE_MAX 0.025
#define LEARNING_RATE_MIN 0.001
#define INITIAL_WEIGHT_FIX 0.5
#define INITIAL_WEIGHT_MIN -1.0
#define INITIAL_WEIGHT_MAX 1.0
#define FILTER_LOW_BOUND 3
#define FILTER_HIGH_RATIO 0.0001

// SECONDARY HYPERPARAMETERS
#define DROPOUT_RATE_MAX 0.2
#define FREQ_BUCKETS 10
#define PREDICTION_MAX 5
#define VOCABULARY_HASH_MAX 30000000
#define MONTE_CARLO_EMERGENCY 100
#define INVALID_INDEX_MAX 10
#define LOG_PERIOD_PASS 1000
#define LOG_PERIOD_CORPUS 500
#define LOG_PERIOD_SENT 100
#define BACKTRACE_DEPTH 10

#define SENTENCE_DELIMITERS ".?!"
#define WORD_DELIMITERS " \t\n\r,:;(){}[]<>\"'’/\\%#$&~*+=^_"

// Number of characters in path
#define PATH_CHARACTER_MAX 255

// Number of characters in line
#define LINE_CHARACTER_MAX 512

// Number of sentences to allocate per file initially
#define SENTENCE_THRESHOLD 64

// Flags
#define FLAG_DEBUG
#define FLAG_LOG
#define FLAG_COLOR_LOG
//#define FLAG_BINARY_INPUT
//#define FLAG_BINARY_OUTPUT
#define FLAG_NEGATIVE_SAMPLING
#define FLAG_UNIGRAM_DISTRIBUTION
#define FLAG_CALCULATE_LOSS
#define FLAG_BACKUP_VOCABULARY
#define FLAG_BACKUP_WEIGHTS
#define FLAG_PRINT_WORD_ERRORS
#define FLAG_PRINT_INDEX_ERRORS
#define FLAG_FILTER_VOCABULARY_STOP
#define FLAG_FILTER_VOCABULARY_LOW
#define FLAG_FILTER_VOCABULARY_HIGH
#define FLAG_TEST_SIMILARITY
//#define FLAG_TEST_CONTEXT
#define FLAG_TEST_ORTHANT
#define FLAT_DISTANCE_COSINE
//#define FLAG_SENT
//#define FLAG_STEM
//#define FLAG_MONTE_CARLO
#define FLAG_FIXED_LEARNING_RATE
//#define FLAG_FIXED_INITIAL_WEIGHTS
//#define FLAG_DROPOUT
//#define FLAG_FREE_MEMORY

#ifdef FLAG_LOG
#define FLAG_LOG_FILE
#endif

#ifdef FLAG_DEBUG
#undef FLAG_LOG_FILE
#define flog stdout
#endif

// Paths
#define TRAIN_PATH arg_train
#define TEST_PATH arg_test
#define STOP_PATH arg_stop
#define VOCABULARY_PATH "out/vocab.tsv"
#define WEIGHTS_IH_PATH "out/weights-ih.tsv"
#define WEIGHTS_HO_PATH "out/weights-ho.tsv"
#define LOG_PATH "out/log.txt"
#define SENT_PATH "out/sent.tsv"
#define FILTER_PATH "out/filter.tsv"

// Messages
#define ERROR_FILE "File error occurred"
#define ERROR_MEMORY "Memory error occurred"
#define ERROR_COMMAND "Command error occurred"
#define ERROR_CMDARGS "Command line arguments error occurred"

// Shortcuts
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define limit_norm(x, a, b, m, s) ({dt_float _x = (x) * (s) + (m); _x > a && _x < b ? _x : m;})
#define random_unif(a, b) ((rand() / (dt_float) RAND_MAX) * (b - a) + a)
#define random_norm(a, b) ({dt_float _m = (a + (b - a) * 0.5); limit_norm(sqrt(-2.0 * log(random_unif(0.0, 1.0))) * cos(2.0 * M_PI * random_unif(0.0, 1.0)), a, b, _m, _m * 0.3);})
#define random(a, b) random_unif(a, b)
#define random_int(a, b) ((dt_int) min(max(a, random(a, b)), b))
#ifdef FLAG_LOG
#define memcheck(ptr) memcheck_log(ptr, __FILE__, __func__, __LINE__)
typedef enum eColor { GRAY, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, NONE } eColor;
#define echo(...) echo_color(NONE, 0, __VA_ARGS__)
#define echo_info(...) echo_color(YELLOW, 0, "INFO: " __VA_ARGS__)
#define echo_succ(...) echo_color(GREEN, 0, "SUCCESS: " __VA_ARGS__)
#define echo_fail(...) echo_color(RED, 0, "FAIL: " __VA_ARGS__)
#define echo_cond(ok, ...) (ok ? echo_succ(__VA_ARGS__) : echo_fail(__VA_ARGS__))
#define echo_repl(...) echo_color(NONE, 1, __VA_ARGS__)
#ifdef FLAG_DEBUG
#define debug(...) echo_color(MAGENTA, 0, "DEBUG: " __VA_ARGS__)
#endif
#else
#define memcheck(ptr)
#endif
#define DEF_LINE(x) static dt_int (x) = __LINE__
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
	dt_uint context_max;
	dt_int freq;
	dt_float freq_dist;
	dt_float prob;
	dt_float dist;
	struct xWord* left;
	struct xWord* right;
	struct xWord* next;
	struct xWord** target;
	struct xContext* context;
} xWord;

typedef struct xContext {
	xWord* word;
	struct xContext* left;
	struct xContext* right;
} xContext;

typedef struct xSent {
	dt_char* sent;
	dt_float* vec;
	dt_float dist;
} xSent;

typedef union xBit {
	dt_uint on : 1;
} xBit;

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
void sentences_encode();
void sentences_similarity();

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
#ifdef FLAG_LOG
#include "log.h"
#endif

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
