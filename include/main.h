#ifndef H_MAIN_INCLUDE
#define H_MAIN_INCLUDE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <ctype.h>

#define WINDOW_MAX 2

// Number of sentences in file
#define SENTENCE_MAX 10

// Number of words in sentence
#define WORD_MAX 100

// Number of characters in word
#define CHARACTER_MAX 50

#define LOG_FILE stdout

typedef struct xWord {
	char word[CHARACTER_MAX];
	unsigned int count;
	struct xWord* left;
	struct xWord* right;
} xWord;

typedef struct xBit {
	unsigned int on : 1;
} xBit;

#endif
