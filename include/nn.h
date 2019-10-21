// http://www.cs.bham.ac.uk/~jxb/INC/nn.html

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

#define random() ((double)rand() / ((double) RAND_MAX + 1))

void start_training();

#endif
