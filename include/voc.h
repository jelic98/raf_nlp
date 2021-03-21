#ifndef H_VOC_INCLUDE
#define H_VOC_INCLUDE

#include "lib.h"

xWord* stops;
xWord** vocab;
dt_int* vocab_hash;
dt_int invalid_index[INVALID_INDEX_MAX];
dt_int invalid_index_last;

#ifdef FLAG_NEGATIVE_SAMPLING
dt_int corpus_freq_sum, corpus_freq_max;
#ifndef FLAG_MONTE_CARLO
xWord** samples;
#endif
#endif

dt_int cmp_int(const void*, const void*);
dt_int cmp_freq(const void*, const void*);
dt_int cmp_freq_dist(const void*, const void*);
dt_int word_to_index(xWord**, const dt_char*);
void word_lower(dt_char*);
void word_clean(dt_char*, dt_int*);
dt_int word_stop(const dt_char*);
void vocab_filter(xWord*);
void vocab_save(xWord**);
void vocab_map(xWord**);
void vocab_target(xWord**);
void vocab_freq(xWord**, dt_int*, dt_int*);
void vocab_sample(xWord**);
xWord* index_to_word(xWord**, dt_int);
dt_int index_valid(dt_int);
void invalid_index_print();
void calculate_distribution();

#ifdef H_VOC_IMPLEMENT
// Compare two integers
dt_int cmp_int(const void* a, const void* b) {
	return *(dt_int*) a - *(dt_int*) b;
}

#ifdef FLAG_FILTER_VOCABULARY_HIGH
// Compare two words by their frequency
dt_int cmp_freq(const void* a, const void* b) {
	return (*(xWord**) a)->freq - (*(xWord**) b)->freq;
}
#endif

#ifdef FLAG_NEGATIVE_SAMPLING
#ifndef FLAG_MONTE_CARLO
#ifndef FLAG_UNIGRAM_DISTRIBUTION
// Compare two words by their normalized frequency
dt_int cmp_freq_dist(const void* a, const void* b) {
	dt_float diff = (*(xWord**) a)->freq_dist - (*(xWord**) b)->freq_dist;
	return diff > 0 ? 1 : diff < 0 ? -1 : 0;
}
#endif
#endif
#endif

// Get vocabulary index by word hash
dt_int word_to_index(xWord** vocab, const dt_char* word) {
	xWord** ptr = map_get(vocab, &vocab_hash, word);

	if(ptr) {
		dt_int index = (*ptr)->index;

		if(index >= 0 && index < pattern_max) {
			return index;
		}
	}

#ifdef FLAG_LOG
#ifdef FLAG_PRINT_WORD_ERRORS
	echo_fail("%s not found in corpus", word);
#endif
#endif

	return -1;
}

// Convert word to lowercase
void word_lower(dt_char* word) {
	while(*word) {
		*word = tolower(*word);
		word++;
	}
}

// Strip whitespaces from word
void word_clean(dt_char* word, dt_int* sent_end) {
	word_lower(word);

	dt_int end = strlen(word) - 1;

	*sent_end = strchr(SENTENCE_DELIMITERS, word[end]) != NULL;

	if(*sent_end) {
		word[end] = '\0';
	}

#ifdef FLAG_STEM
	word[stem(word, 0, strlen(word) - 1) + 1] = '\0';
#endif
}

#ifdef FLAG_FILTER_VOCABULARY_STOP
// Check if word is stop words
dt_int word_stop(const dt_char* word) {
	if(strlen(word) < 3) {
		return 1;
	}

	const dt_char* p;
	for(p = word; *p && (isalpha(*p) || *p == '-'); p++)
		;

	return *p || list_contains(stops, word);
}
#endif

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
// Filter vocabulary by word frequency
void vocab_filter(xWord* corpus) {
	xWord** vocab = (xWord**) calloc(pattern_max, sizeof(xWord*));
	memcheck(vocab);

	dt_int p, index = 0;
	bst_flatten(corpus, vocab, &index);

	for(filter_max = p = 0; p < pattern_max; p++) {
		vocab[p]->index = 0;
	}

#ifdef FLAG_FILTER_VOCABULARY_HIGH
	dt_int filter_high = filter_max += output_max * FILTER_HIGH_RATIO;
#endif

#ifdef FLAG_FILTER_VOCABULARY_LOW
	dt_int filter_low;

	for(filter_low = p = 0; p < pattern_max; p++) {
		filter_low += vocab[p]->freq < FILTER_LOW_BOUND;
	}

	filter_max += filter_low;
#endif

	if(filter_max > 0) {
		filter = (xWord**) calloc(filter_max, sizeof(xWord*));
		memcheck(filter);
		index = 0;
	}

#ifdef FLAG_FILTER_VOCABULARY_HIGH
	qsort(vocab, pattern_max, sizeof(xWord*), cmp_freq);

	dt_int old_pattern_max1 = pattern_max;
	pattern_max = input_max = output_max -= filter_high;

	for(p = pattern_max; p < old_pattern_max1; p++) {
		vocab[p]->freq *= -1;
		filter[index++] = vocab[p];
	}
#endif

#ifdef FLAG_FILTER_VOCABULARY_LOW
	dt_int old_pattern_max2 = pattern_max;
	pattern_max = input_max = output_max -= filter_low;

	for(p = 0; p < old_pattern_max2; p++) {
		if(vocab[p]->freq < FILTER_LOW_BOUND) {
			vocab[p]->freq *= -1;
			filter[index++] = vocab[p];
		}
	}
#endif

	free(vocab);
}
#endif

#ifdef FLAG_BACKUP_VOCABULARY
// Save vocabulary to file
void vocab_save(xWord** vocab) {
	FILE* fvoc = fopen(VOCABULARY_PATH, "w");

	if(!fvoc) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
		return;
	}

	dt_int p;

	for(p = 0; p < pattern_max; p++) {
		fprintf(fvoc, "%s\t%d\n", vocab[p]->word, vocab[p]->freq);
	}

	if(fclose(fvoc) == EOF) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
	}

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
	FILE* ffil = fopen(FILTER_PATH, "w");

	if(!ffil) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
		return;
	}

	dt_int f;

	for(f = 0; f < filter_max; f++) {
		fprintf(ffil, "%s\t%d\n", filter[f]->word, -filter[f]->freq);
	}

	if(fclose(ffil) == EOF) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
	}
#endif
}
#endif

// Convert vocabulary BST to hash map
void vocab_map(xWord** vocab) {
	dt_int p;

	map_init(vocab, &vocab_hash, pattern_max);

	for(p = 0; p < pattern_max; p++) {
		*map_get(vocab, &vocab_hash, vocab[p]->word) = vocab[p];
	}
}

// Create target words at output layer
void vocab_target(xWord** vocab) {
	dt_int p;

	for(p = 0; p < pattern_max; p++) {
		vocab[p]->target = (xWord**) calloc(vocab[p]->context_max, sizeof(xWord*));
		memcheck(vocab[p]->target);
		vocab[p]->target_freq = (dt_int*) calloc(vocab[p]->context_max, sizeof(dt_int));
		memcheck(vocab[p]->target_freq);
		dt_int index = 0;
		context_flatten(vocab[p]->context, vocab[p]->target, vocab[p]->target_freq, &index);
		context_release(vocab[p]->context);
		vocab[p]->context = NULL;
		vocab[p]->context_max = index;
	}
}

#ifdef FLAG_NEGATIVE_SAMPLING
// Get sum and maximum frequency in vocabulary 
void vocab_freq(xWord** vocab, dt_int* sum, dt_int* max) {
	dt_int p;

	for(*sum = *max = p = 0; p < pattern_max; p++) {
		*sum += vocab[p]->freq;
		*max = max(*max, vocab[p]->freq);
	}
}

#ifndef FLAG_MONTE_CARLO
// Create array from which negative samples will be picked
void vocab_sample(xWord** vocab) {
#ifdef FLAG_UNIGRAM_DISTRIBUTION
	samples = (xWord**) calloc(corpus_freq_sum, sizeof(xWord*));
	memcheck(samples);

	dt_int p, c, tmp;

	for(tmp = p = 0; p < pattern_max; p++) {
		for(c = 0; c < vocab[p]->freq; c++) {
			samples[tmp++] = vocab[p];
		}
	}
#else
	xWord** copies = (xWord**) calloc(pattern_max, sizeof(xWord*));
	memcheck(copies);

	dt_int p, ck;

	for(p = 0; p < pattern_max; p++) {
		vocab[p]->freq_dist = pow(vocab[p]->freq, 0.75) / pow(corpus_freq_sum, 0.75);
		copies[p] = vocab[p];
	}

	qsort(copies, pattern_max, sizeof(xWord*), cmp_freq_dist);

	for(p = 0; p < pattern_max; p++) {
		ck = pattern_max / 2 + (p > 0) * p / 2 * (1 + 2 * (p % 2 - 1)) + p % 2 - !(pattern_max % 2);
		samples[ck] = copies[p];
	}

	free(copies);
#endif
}
#endif
#endif

// Get word by vocabulary index
xWord* index_to_word(xWord** vocab, dt_int index) {
	return vocab[index];
}

// Check if vocabulary index is valid
dt_int index_valid(dt_int index) {
	dt_int valid = index >= 0 && index < input_max;

	if(!valid) {
		dt_int i, found = 0;

		for(i = 0; i < invalid_index_last; i++) {
			if(invalid_index[i] == index) {
				found = 1;
				break;
			}
		}

		if(!found) {
			invalid_index[invalid_index_last++] = index;
		}
	}

	return valid;
}

#ifdef FLAG_PRINT_INDEX_ERRORS
// Print all invalid indices referenced during training
void invalid_index_print() {
	if(invalid_index_last > 0) {
		dt_int i;

		for(i = 0; i < invalid_index_last; i++) {
#ifdef FLAG_LOG
			echo_fail("Invalid index %d:\t%d", i + 1, invalid_index[i]);
#endif
		}
	} else {
#ifdef FLAG_LOG
		echo_succ("No invalid indices");
#endif
	}
}
#endif

// Calculate frequency distribution for manual analytics
void calculate_distribution() {
#ifdef FLAG_LOG
	echo("Calculating distribution");
#endif

	dt_int* freqs = (dt_int*) calloc(pattern_max, sizeof(dt_int));
	memcheck(freqs);

	dt_int p;

	for(p = 0; p < pattern_max; p++) {
		freqs[p] = vocab[p]->freq;
	}

	qsort(freqs, pattern_max, sizeof(dt_int), cmp_int);

	dt_int buck, span = pattern_max / FREQ_BUCKETS;

	for(p = 1; p <= FREQ_BUCKETS; p++) {
		buck = p == FREQ_BUCKETS ? pattern_max - 1 : (p - 1) * span;

#ifdef FLAG_LOG
		echo_info("Sample #%d (%d/%d): %d occurrences", p, buck + 1, pattern_max, freqs[buck]);
#endif
	}

	free(freqs);

#ifdef FLAG_LOG
	echo_succ("Done calculating distribution");
#endif
}
#endif
#endif
