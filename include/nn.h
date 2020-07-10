#ifndef H_NN_INCLUDE
#define H_NN_INCLUDE

#include "lib.h"

#define TEST_INDEX "3"

#define EPOCH_MAX 3
#define HIDDEN_MAX 50
#define WINDOW_MAX 5
#define NEGATIVE_SAMPLES_MAX 15
#define LEARNING_RATE_FIX 0.01
#define LEARNING_RATE_MAX 0.025
#define LEARNING_RATE_MIN 0.001
#define INITIAL_WEIGHT_FIX 0.5
#define INITIAL_WEIGHT_MIN -1.0
#define INITIAL_WEIGHT_MAX 1.0
#define DROPOUT_RATE_MAX 0.2
#define FILTER_RATIO 0.05
#define THREAD_MAX 2

#define MONTE_CARLO_EMERGENCY 100
#define INVALID_INDEX_MAX 10
#define LOG_PERIOD_PASS 1000
#define LOG_PERIOD_CORPUS 500
#define BACKTRACE_DEPTH 10

#define SENTENCE_DELIMITERS ".?!"
#define WORD_DELIMITERS " \t\n\r,:;(){}[]<>\"'’/\\%#$&~*+=^_"

// Number of characters in line
#define LINE_CHARACTER_MAX 512

// Number of sentences to allocate per file initially
#define SENTENCE_THRESHOLD 64

// Number of words to allocate per sentence initially
#define WORD_THRESHOLD 16

// Flags
#define FLAG_DEBUG
#define FLAG_LOG
#define FLAG_COLOR_LOG
#define FLAG_NEGATIVE_SAMPLING
#define FLAG_UNIGRAM_DISTRIBUTION
#define FLAG_CALCULATE_LOSS
//#define FLAG_DROPOUT
//#define FLAG_MONTE_CARLO
//#define FLAG_STEM
//#define FLAG_FIXED_LEARNING_RATE
//#define FLAG_FIXED_INITIAL_WEIGHTS
//#define FLAG_FILTER_VOCABULARY
#define FLAG_BACKUP_VOCABULARY
//#define FLAG_BACKUP_WEIGHTS
//#define FLAG_BINARY_INPUT
//#define FLAG_BINARY_OUTPUT
//#define FLAG_INTERACTIVE_MODE
//#define FLAG_PRINT_INDEX_ERRORS

#ifdef FLAG_LOG
#define FLAG_LOG_FILE
#endif

#ifdef FLAG_DEBUG
#undef FLAG_LOG_FILE
#define flog stdout
#endif

// Paths
#define CORPUS_PATH "res/corpus/" TEST_INDEX ".txt"
#define TEST_PATH "res/test/" TEST_INDEX ".txt"
#define STOP_PATH "res/misc/stop.txt"
#define VOCABULARY_PATH "out/vocab.tsv"
#define WEIGHTS_IH_PATH "out/weights-ih.tsv"
#define WEIGHTS_HO_PATH "out/weights-ho.tsv"
#define LOG_PATH "out/log.txt"

// Messages
#define ERROR_FILE "File error occurred"
#define ERROR_MEMORY "Memory error occurred"
#define ERROR_COMMAND "Command error occurred"

// Shortcuts
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define limit_norm(x, a, b, m, s) ({dt_float _x = (x) * (s) + (m); _x > a && _x < b ? _x : m;})
#define random_unif(a, b) ((rand() / (dt_float) RAND_MAX) * (b - a) + a)
#define random_norm(a, b) ({dt_float _m = (a + (b - a) * 0.5); limit_norm(sqrt(-2.0 * log(random_unif(0.0, 1.0))) * cos(2.0 * M_PI * random_unif(0.0, 1.0)), a, b, _m, _m * 0.3);})
#define random(a, b) random_unif(a, b)
#define random_int(a, b) ((dt_int) min(max(a, random(a, b)), b))
#define memcheck(ptr) memcheck_log(ptr, __FILE__, __func__, __LINE__)
#ifdef FLAG_LOG
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
#endif

// Internal data limits
#define DT_FLOAT_MIN DBL_MIN

// Internal data types
typedef char dt_char;
typedef int dt_int;
typedef unsigned int dt_uint;
typedef double dt_float;

typedef struct xWord {
	dt_char* word;
	dt_int index;
	dt_uint context_max;
	dt_uint freq;
	dt_float freq_dist;
	dt_float prob;
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
void sentence_encode(dt_char*, dt_float*);

// Dependencies
#ifdef FLAG_STEM
#define H_STMR_IMPLEMENT
#include "stmr.h"
#endif
#endif
