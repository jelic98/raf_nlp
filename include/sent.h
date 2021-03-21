#ifndef H_SENT_INCLUDE
#define H_SENT_INCLUDE

#include "lib.h"

dt_int cmp_sent_dist(const void*, const void*);
void sentence_encode(dt_char*, dt_float*, dt_int, xWord**, dt_float**);
void sentences_encode(dt_int, xWord**, dt_float**);
void sentences_similarity(dt_int);

#ifdef H_SENT_IMPLEMENT
// Compare two sentences by their distance to target sentence
dt_int cmp_sent_dist(const void* a, const void* b) {
	dt_float diff = (*(xSent**) b)->dist - (*(xSent**) a)->dist;
	return diff > 0 ? 1 : diff < 0 ? -1 : 0;
}

// Encode single sentence using generated word embeddings
void sentence_encode(dt_char* sentence, dt_float* vector, dt_int vec_len, xWord** vocab, dt_float** vectors) {
	memset(vector, 0, vec_len * sizeof(dt_float));

	dt_int index, sent_end, j;
	const dt_char* sep = WORD_DELIMITERS;
	dt_char* tok = strtok(sentence, sep);

	while(tok) {
		word_clean(tok, &sent_end);

		dt_int skip = 0;

#ifdef FLAG_FILTER_VOCABULARY_STOP
		skip = skip || word_stop(tok);
#endif

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
		skip = skip || filter_contains(filter, tok);
#endif

		if(!skip) {
			index = word_to_index(vocab, tok);

			if(index_valid(index)) {
				for(j = 0; j < vec_len; j++) {
					vector[j] += vectors[index][j];
				}
			}
		}

		tok = strtok(NULL, sep);
	}

	vector_normalize(vector, vec_len);
}

// Encode sentences from sentences file
void sentences_encode(dt_int vec_len, xWord** vocab, dt_float** vectors) {
#ifdef FLAG_LOG
	struct timespec time_local;
	clock_gettime(CLOCK_MONOTONIC, &time_local);
	echo("Started sentences encoding");
#endif

	FILE* fsentin = fopen(TRAIN_PATH, "r");

#ifdef FLAG_BINARY_OUTPUT
	FILE* fsentout = fopen(SENT_PATH, "wb");
#else
	FILE* fsentout = fopen(SENT_PATH, "w");
#endif

	if(!fsentin || !fsentout) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
		return;
	}

	dt_char line[LINE_CHARACTER_MAX];
	dt_int sent_end;
	dt_float vec[vec_len];

#ifdef FLAG_LOG
	dt_int index = 0;
#endif

#ifndef FLAG_BINARY_OUTPUT
	dt_int j;
#endif

	while(fgets(line, LINE_CHARACTER_MAX, fsentin)) {
		line[strlen(line) - 1] = '\0';
		word_clean(line, &sent_end);
		sentence_encode(line, vec, vec_len, vocab, vectors);

#ifdef FLAG_LOG
		if(!(++index % LOG_PERIOD_SENT)) {
			echo("Encoded sentences: %d", index);
		}
#endif

#ifdef FLAG_BINARY_OUTPUT
		fwrite(vec, sizeof(dt_float), vec_len, fsentout);
#else
		for(j = 0; j < vec_len; j++) {
			fprintf(fsentout, j ? "\t%lf" : "%lf", vec[j]);
		}

		fprintf(fsentout, "\n");
#endif
	}

	if(fclose(fsentin) == EOF || fclose(fsentout) == EOF) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
	}

#ifdef FLAG_LOG
	echo_succ("Finished sentences encoding (%d sec)", time_get(time_local));
#endif
}

// Perform similarity test for sentences in sentences file
void sentences_similarity(dt_int vec_len) {
#ifdef FLAG_LOG
	struct timespec time_local;
	clock_gettime(CLOCK_MONOTONIC, &time_local);
	echo("Started sentence similarity testing");
#endif

	FILE* fsentin = fopen(TRAIN_PATH, "r");

#ifdef FLAG_BINARY_OUTPUT
	FILE* fsentout = fopen(SENT_PATH, "rb");
#else
	FILE* fsentout = fopen(SENT_PATH, "r");
#endif

	if(!fsentin || !fsentout) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
		return;
	}

	dt_char line[LINE_CHARACTER_MAX];
	dt_int sent_end, s, sentences_max, prediction_max, index = -1;
#ifndef FLAG_BINARY_OUTPUT
	dt_int j;
#endif

	xSent** sent = (xSent**) calloc(SENTENCE_THRESHOLD, sizeof(xSent*));
	xSent** dist = (xSent**) calloc(SENTENCE_THRESHOLD, sizeof(xSent*));

	while(index++, fgets(line, LINE_CHARACTER_MAX, fsentin)) {
		line[strlen(line) - 1] = '\0';
		word_clean(line, &sent_end);

		dist[index] = sent[index] = (xSent*) calloc(1, sizeof(xSent));

		sent[index]->sent = (dt_char*) calloc(strlen(line) + 1, sizeof(dt_char));
		strcpy(sent[index]->sent, line);

		sent[index]->vec = (dt_float*) calloc(vec_len, sizeof(dt_float));

#ifdef FLAG_BINARY_OUTPUT
		fread(sent[index]->vec, sizeof(dt_float), vec_len, fsentout);
#else
		for(j = 0; j < vec_len; j++) {
			fscanf(fsentout, "%lf", &sent[index]->vec[j]);
		}
#endif
	}

	prediction_max = min(index - 1, PREDICTION_MAX);

	for(sentences_max = index, index = 0; index < sentences_max; index++) {
#ifdef FLAG_LOG
		echo_info("Sentence: %s", sent[index]->sent);
#endif

		for(s = 0; s < sentences_max; s++) {
			vector_distance(sent[s]->vec, sent[index]->vec, vec_len, &sent[index]->dist);
		}

		qsort(dist, sentences_max, sizeof(xSent*), cmp_sent_dist);

		for(s = 1; s <= prediction_max; s++) {
#ifdef FLAG_LOG
			echo("#%d\t%lf\t%s", s, dist[s]->dist, dist[s]->sent);
#endif
		}
	}

	if(fclose(fsentin) == EOF || fclose(fsentout) == EOF) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
	}

#ifdef FLAG_LOG
	echo_succ("Finished sentence similarity testing (%d sec)", time_get(time_local));
#endif
}
#endif
#endif
