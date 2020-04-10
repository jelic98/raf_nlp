#ifndef H_NN_INCLUDE
#define H_NN_INCLUDE

#include "lib.h"

#define EPOCH_MAX 1
#define HIDDEN_MAX 50
#define WINDOW_MAX 5
#define NEGATIVE_SAMPLES_MAX 6
#define INITIAL_WEIGHT_MAX 0.5
#define LEARNING_RATE_MAX 0.025
#define LEARNING_RATE_MIN 0.001

#define MONTE_CARLO_FACTOR 1.0
#define MONTE_CARLO_EMERGENCY 10
#define INVALID_INDEX_MAX 10
#define LOG_PERIOD_PASS 100
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
#define FLAG_LOG
#define FLAG_DEBUG
#define FLAG_NEGATIVE_SAMPLING
//#define FLAG_STEM
//#define FLAG_BACKUP_WEIGHTS
//#define FLAG_PRINT_CORPUS
//#define FLAG_PRINT_INDEX_ERRORS

#ifdef FLAG_LOG
#define FLAG_LOG_FILE
#endif

#ifdef FLAG_DEBUG
#undef FLAG_LOG_FILE
#define flog stdout
#endif

// Paths
#define CORPUS_PATH "res/corpus/2.txt"
#define TEST_PATH "res/test/2.txt"
#define STOP_PATH "res/misc/stop.txt"
#define WEIGHTS_IH_PATH "out/weights-ih.txt"
#define WEIGHTS_HO_PATH "out/weights-ho.txt"
#define LOG_PATH "out/log.txt"

// Messages
#define ERROR_FILE "File error occurred"
#define ERROR_MEMORY "Memory error occurred"

// Shortcuts
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define random(a, b) ((rand() / (dt_float) RAND_MAX) * (b - a) + a)
#define memcheck(ptr) memcheck_log(ptr, __FILE__, __func__, __LINE__)
#ifdef FLAG_LOG
typedef enum eColor { GRAY, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, NONE } eColor;
#define echo(...) echo_color(NONE, __VA_ARGS__)
#define echo_info(...) echo_color(YELLOW, "INFO: " __VA_ARGS__)
#define echo_succ(...) echo_color(GREEN, "SUCCESS: " __VA_ARGS__)
#define echo_fail(...) echo_color(RED, "FAIL: " __VA_ARGS__)
#define echo_cond(ok, ...) (ok ? echo_succ(__VA_ARGS__) : echo_fail(__VA_ARGS__))
#ifdef FLAG_DEBUG
#define debug(...) echo_color(MAGENTA, "DEBUG: " __VA_ARGS__)
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
	dt_uint freq;
	dt_uint context_max;
	dt_float prob;
	dt_int index;
	struct xWord** target;
	struct xWord* left;
	struct xWord* right;
	struct xWord* next;
	struct xContext* context;
} xWord;

typedef struct xContext {
	xWord* word;
	struct xContext* next;
} xContext;

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

// Dependencies
#ifdef FLAG_STEM
#define H_STMR_IMPLEMENT
#include "stmr.h"
#endif
#endif
