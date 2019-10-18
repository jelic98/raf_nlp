#ifndef H_DATA_INCLUDE
#define H_DATA_INCLUDE

#include "include/main.h"

#define WORD_MAX 50
#define WORDS_MAX 100

#define CORPUS_IN_FILE "data/corpus_in.txt"
#define CORPUS_OUT_FILE "data/corpus_out.txt"

#define FILE_ERROR_MESSAGE "File error occurred\n"

void text_to_sentences();
void sentences_to_words();
void words_to_onehot();

#endif
