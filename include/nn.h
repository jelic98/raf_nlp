// http://www.cs.bham.ac.uk/~jxb/INC/nn.html

#ifndef H_NN_INCLUDE
#define H_NN_INCLUDE

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>

#define PATTER_MAX 4
#define INPUT_MAX 2
#define HIDDEN_MAX 2
#define OUTPUT_MAX 1

#define LEARNING_RATE 0.5
#define MOMENTUM 0.9
#define INITIAL_WEIGHT_MAX 0.5

#define EPOCH_MAX 10000
#define ERROR_MAX 0.0004

#define LOG_PERIOD 100
#define OUT stdout

#define random() ((double)rand() / ((double) RAND_MAX + 1))

void start_training();

#endif
