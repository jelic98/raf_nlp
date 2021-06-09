#ifndef H_VOC_INCLUDE
#define H_VOC_INCLUDE

#include "lib.h"

xWord* stops;
xWord** vocab;
dt_ull* vocab_hash;
dt_ull invalid_index[INVALID_INDEX_MAX];
dt_ull invalid_index_last;
dt_ull corpus_freq_sum, corpus_freq_max;
dt_int* samples;

dt_int cmp_ull(const void*, const void*);
dt_int cmp_freq(const void*, const void*);
dt_int cmp_int(const void*, const void*);
dt_ull word_to_index(xWord**, const dt_char*);
void word_lower(dt_char*);
void word_clean(dt_char*, dt_int*);
dt_int word_stop(const dt_char*);
void vocab_filter(xWord*);
void vocab_save(xWord**);
void vocab_map(xWord**);
void vocab_target(xWord**);
void vocab_freq(xWord**, dt_ull*, dt_ull*);
void vocab_sample(xWord**);
xWord* index_to_word(xWord**, dt_ull);
dt_int index_valid(dt_ull);
void invalid_index_print();
void calculate_distribution();

#ifdef H_VOC_IMPLEMENT
// Compare two integers
dt_int cmp_ull(const void* a, const void* b) {
	dt_ull ulla = (*(dt_ull*) a);
	dt_ull ullb = (*(dt_ull*) b);
	dt_int ra = ulla > ullb;
	dt_int rb = ulla < ullb;
	return rb - ra;
}

// Compare two words by their frequency
dt_int cmp_freq(const void* a, const void* b) {
	return (*(xWord**) a)->freq - (*(xWord**) b)->freq;
}

// Get vocabulary index by word hash
dt_ull word_to_index(xWord** vocab, const dt_char* word) {
	xWord** ptr = map_get(vocab, &vocab_hash, word);

	if(ptr) {
		dt_int index = (*ptr)->index;

		if(index >= 0 && index < pattern_max) {
			return index;
		}
	}

	echo_fail("%s not found in corpus", word);

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
		vocab[p]->index = -1;
		filter[index++] = vocab[p];
	}
#endif

#ifdef FLAG_FILTER_VOCABULARY_LOW
	dt_int old_pattern_max2 = pattern_max;
	pattern_max = input_max = output_max -= filter_low;

	for(p = 0; p < old_pattern_max2; p++) {
		if(vocab[p]->freq < FILTER_LOW_BOUND) {
			vocab[p]->index = -1;
			filter[index++] = vocab[p];
		}
	}
#endif

	free(vocab);
}
#endif

// Save vocabulary to file
void vocab_save(xWord** vocab) {
	FILE* fvoc = fopen(VOCABULARY_PATH, "w");

	if(!fvoc) {
		echo_fail(ERROR_FILE);
		return;
	}

	dt_int p;

	for(p = 0; p < pattern_max; p++) {
		fprintf(fvoc, "%s\t%llu\n", vocab[p]->word, vocab[p]->freq);
	}

	if(fclose(fvoc) == EOF) {
		echo_fail(ERROR_FILE);
	}

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
	FILE* ffil = fopen(FILTER_PATH, "w");

	if(!ffil) {
		echo_fail(ERROR_FILE);
		return;
	}

	dt_int f;

	for(f = 0; f < filter_max; f++) {
		fprintf(ffil, "%s\t%llu\n", filter[f]->word, filter[f]->freq);
	}

	if(fclose(ffil) == EOF) {
		echo_fail(ERROR_FILE);
	}
#endif
}

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
		vocab[p]->target_freq = (dt_ull*) calloc(vocab[p]->context_max, sizeof(dt_ull));
		memcheck(vocab[p]->target_freq);
		dt_int index = 0;
		context_flatten(vocab[p]->context, vocab[p]->target, vocab[p]->target_freq, &index);
		context_release(vocab[p]->context);
		vocab[p]->context = NULL;
		vocab[p]->context_max = index;
	}
}

// Get sum and maximum frequency in vocabulary
void vocab_freq(xWord** vocab, dt_ull* sum, dt_ull* max) {
	dt_int p;

	for(*sum = *max = p = 0; p < pattern_max; p++) {
		*sum += vocab[p]->freq;
		*max = max(*max, vocab[p]->freq);
	}
}

// Create array from which negative samples will be picked
void vocab_sample(xWord** vocab) {
	dt_int p, q;

#ifdef FLAG_UNIGRAM_SAMPLING
	dt_int tmp = 0;

	for(p = 0; p < pattern_max; p++) {
		tmp += vocab[p]->freq;
	}

	samples = (dt_int*) calloc(tmp, sizeof(dt_int));
	memcheck(samples);

	for(tmp = p = 0; p < pattern_max; p++) {
		for(q = 0; q < vocab[p]->freq; q++) {
			samples[tmp++] = q;
		}
	}
#else
	samples = (dt_int*) calloc(pattern_max, sizeof(dt_int));
	memcheck(samples);

	dt_ull* freqs = (dt_ull*) calloc(pattern_max, sizeof(dt_ull));
	memcheck(freqs);

	for(p = 0; p < pattern_max; p++) {
		freqs[p] = vocab[p]->freq;
	}

	qsort(freqs, pattern_max, sizeof(dt_ull), cmp_ull);

	dt_int center = pattern_max / 2 + pattern_max % 2 - 1;
	dt_int offset = 0;
	dt_int right = 1;

	for(p = 0; p < pattern_max; p++) {
		q = center;

		if(right) {
			q += offset;
			samples[q] = freqs[p];
			offset += 1;
		} else {
			q -= offset;

			if(q < 0) {
				q += pattern_max;
			}

			samples[q] = freqs[p];
		}

		right = !right;
	}

	free(freqs);
#endif
}

// Get word by vocabulary index
xWord* index_to_word(xWord** vocab, dt_ull index) {
	return vocab[index];
}

// Check if vocabulary index is valid
dt_int index_valid(dt_ull index) {
	dt_int valid = index >= 0 && index < pattern_max;

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

// Print all invalid indices referenced during training
void invalid_index_print() {
	if(invalid_index_last > 0) {
		dt_ull i;

		for(i = 0; i < invalid_index_last; i++) {
			echo_fail("Invalid index %d:\t%d", i + 1, invalid_index[i]);
		}
	} else {
		echo_succ("No invalid indices");
	}
}

// Calculate frequency distribution for manual analytics
void calculate_distribution() {
	echo("Calculating distribution");

	dt_ull* freqs = (dt_ull*) calloc(pattern_max, sizeof(dt_ull));
	memcheck(freqs);

	dt_int p;

	for(p = 0; p < pattern_max; p++) {
		freqs[p] = vocab[p]->freq;
	}

	qsort(freqs, pattern_max, sizeof(dt_ull), cmp_ull);

	dt_int buck, span = pattern_max / FREQ_BUCKETS;

	for(p = 1; p <= FREQ_BUCKETS; p++) {
		buck = p == FREQ_BUCKETS ? pattern_max - 1 : (p - 1) * span;
		echo_info("Sample #%d (%d/%d): %llu occurrences", p, buck + 1, pattern_max, freqs[buck]);
	}

	free(freqs);

	echo_succ("Done calculating distribution");
}
#endif
#endif
