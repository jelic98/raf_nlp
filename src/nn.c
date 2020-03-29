#include "include/nn.h"

static clock_t elapsed_time;

static dt_int epoch;
static dt_float alpha, sum, loss;

static dt_int input_max, hidden_max, output_max, pattern_max;
static dt_int i, j, k, c;
static dt_int p, p1, p2;

static xBit* input;
static dt_float* hidden;
static dt_float* output;
static dt_float* output_raw;
static dt_float** w_ih;
static dt_float** w_ho;
static dt_float* error;
static dt_int** target;
static dt_int* patterns;

static dt_char*** context;
static dt_int* context_total_words;
static dt_int context_total_sentences;

static xWord* vocab;

static dt_int* onehot;
static dt_char* test_word;

static dt_int invalid_index[INVALID_INDEX_MAX];
static dt_int invalid_index_last;

#ifdef FLAG_NEGATIVE_SAMPLING
static dt_int vocab_freq_sum, vocab_freq_max;
static dt_int** samples;
#endif

static FILE* ffilter;

#ifdef FLAG_LOG
static FILE* flog;
#endif

#ifdef FLAG_DEBUG
static void screen_clear() {
	printf("\e[1;1H\e[2J");
}
#endif

#ifdef FLAG_DEBUG
static dt_float time_get(clock_t start) {
	return (dt_float)(clock() - start) / CLOCKS_PER_SEC;
}
#endif

static dt_int* map_get(const dt_char* word) {
	dt_uint h = 0;

	for(size_t i = 0; word[i]; i++) {
		h = (h << 3) + (h >> (sizeof(h) * sizeof(dt_char) - 3)) + word[i];
	}

	return &onehot[h % pattern_max];
}

static xWord* bst_get(xWord* node, dt_int* index) {
	if(node) {
		xWord* word = bst_get(node->left, index);

		if(word) {
			return word;
		}

		if(!(*index)--) {
			return node;
		}

		word = bst_get(node->right, index);

		if(word) {
			return word;
		}
	}

	return NULL;
}

static xWord* bst_insert(xWord* node, const dt_char* word, dt_int* success) {
	if(!node) {
		node = (xWord*) calloc(1, sizeof(xWord));
		node->word = word;
		node->freq = 1;
		node->context_count = 0;
		node->prob = 0.0;
		node->left = node->right = NULL;
		*success = 1;
		return node;
	}

	dt_int cmp = strcmp(word, node->word);

	if(cmp < 0) {
		node->left = bst_insert(node->left, word, success);
	} else if(cmp > 0) {
		node->right = bst_insert(node->right, word, success);
	} else {
		node->freq++;
	}

	return node;
}

static void bst_to_map(xWord* node, dt_int* index) {
	if(node) {
		bst_to_map(node->left, index);
		*map_get(node->word) = (*index)++;
		bst_to_map(node->right, index);
	}
}

#ifdef FLAG_NEGATIVE_SAMPLING
static void bst_freq_sum(xWord* node, dt_int* sum) {
	if(node) {
		*sum += node->freq;
		bst_freq_sum(node->left, sum);
		bst_freq_sum(node->right, sum);
	}
}

static void bst_freq_max(xWord* node, dt_int* max) {
	if(node) {
		*max = max(*max, node->freq);
		bst_freq_max(node->left, max);
		bst_freq_max(node->right, max);
	}
}
#endif

#ifdef FLAG_PRINT_VOCAB
static void bst_print(xWord* node, dt_int* index) {
	if(node) {
		bst_print(node->left, index);
		printf("Vocab #%d:\t%s\n", ++(*index), node->word);
		bst_print(node->right, index);
	}
}
#endif

static void bst_release(xWord* node) {
	if(node) {
		bst_release(node->left);
		bst_release(node->right);
		node->left = NULL;
		node->right = NULL;
		free(node);
		node->context_count = 0;
		node = NULL;
	}
}

static xWord* index_to_word(dt_int index) {
	return bst_get(vocab, &index);
}

static dt_int index_valid(dt_int index) {
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

#ifdef FLAG_DEBUG
#ifdef FLAG_PRINT_ERRORS
static void invalid_index_print() {
	if(invalid_index_last > 0) {
		dt_int i;

		for(i = 0; i < invalid_index_last; i++) {
			printf("Invalid index %d:\t%d\n", i + 1, invalid_index[i]);
		}
	} else {
		printf("No invalid indices\n");
	}
}
#endif
#endif

static dt_int word_to_index(const dt_char* word) {
	dt_int index = *map_get(word);

	return index < pattern_max ? index : -1;
}

static void word_lower(dt_char* word) {
	while(*word) {
		*word = tolower(*word);
		word++;
	}
}

static dt_int word_filter(dt_char* word) {
	if(strlen(word) < 2) {
		return 1;
	}

	if(!ffilter) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
#endif
		return 0;
	}

	dt_char line[LINE_CHARACTER_MAX];

	fseek(ffilter, 0, SEEK_SET);

	while(fgets(line, LINE_CHARACTER_MAX, ffilter)) {
		line[strlen(line) - 1] = '\0';

		if(!strcmp(line, word)) {
			return 1;
		}
	}

	return 0;
}

static dt_int word_end(dt_char* word) {
	word_lower(word);

	dt_int end = strlen(word) - 1;

	if(strchr(SENTENCE_DELIMITERS, word[end])) {
		word[end] = '\0';
		return 1;
	}

	return 0;
}

static void onehot_set(xBit* onehot, dt_int index, dt_int size) {
	dt_int q;

	for(q = 0; q < size; q++) {
		onehot[q].on = 0;
	}

	onehot[p = index].on = 1;
}

static dt_int contains_context(xWord* center, dt_int center_index, dt_int context) {
	for(c = 0; c < center->context_count; c++) {
		if(target[center_index][c] == context) {
			return 1;
		}
	}

	return 0;
}

static void parse_corpus() {
	FILE* fin = fopen(CORPUS_PATH, "r");

	if(!fin) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
#endif
		return;
	}

	dt_int i = -1, j = -1, end = 0, success;
	dt_char line[LINE_CHARACTER_MAX];
	dt_char* sep = WORD_DELIMITERS;
	dt_char* tok;

	context = (dt_char***) calloc(SENTENCE_THRESHOLD, sizeof(dt_char**));
	context_total_words = (dt_int*) calloc(SENTENCE_THRESHOLD, sizeof(dt_int));

	while(fgets(line, LINE_CHARACTER_MAX, fin)) {
		tok = strtok(line, sep);

		while(tok) {
			if(i == -1 || end) {
				if(i > 0 && !((i + 1) % SENTENCE_THRESHOLD)) {
					context = (dt_char***) realloc(context, (i + SENTENCE_THRESHOLD) * sizeof(dt_char**));
					context_total_words =
					    (dt_int*) realloc(context_total_words, (i + SENTENCE_THRESHOLD) * sizeof(dt_int));
				} else {
					context_total_sentences = (j = -1, ++i + 1);
					context[i] = (dt_char**) calloc(WORD_THRESHOLD, sizeof(dt_char*));
				}
			}

			end = word_end(tok);

			if(!word_filter(tok)) {
				if(j > 0 && !((j + 1) % WORD_THRESHOLD)) {
					context[i] = (dt_char**) realloc(context[i], (j + WORD_THRESHOLD) * sizeof(dt_char*));
				}

				context_total_words[i] = ++j + 1;
				strcpy(context[i][j] = (dt_char*) calloc(strlen(tok) + 1, sizeof(dt_char)), tok);
				success = 0;
				vocab = bst_insert(vocab, context[i][j], &success);

				if(success) {
					pattern_max = input_max = ++output_max;
					hidden_max = HIDDEN_MAX;
				}
			}

			tok = strtok(NULL, sep);
		}
	}

	if(fclose(fin) == EOF) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
#endif
	}
}

static void resources_allocate() {
	onehot = (dt_int*) calloc(pattern_max, sizeof(dt_int));
	input = (xBit*) calloc(input_max, sizeof(xBit));
	hidden = (dt_float*) calloc(hidden_max, sizeof(dt_float));
	output = (dt_float*) calloc(output_max, sizeof(dt_float));
	output_raw = (dt_float*) calloc(output_max, sizeof(dt_float));

	target = (dt_int**) calloc(pattern_max, sizeof(dt_int*));
	for(p = 0; p < pattern_max; p++) {
		target[p] = (dt_int*) calloc(output_max, sizeof(dt_int));
		memset(target[p], -1, output_max * sizeof(dt_int));
	}

	w_ih = (dt_float**) calloc(input_max, sizeof(dt_float*));
	for(i = 0; i < input_max; i++) {
		w_ih[i] = (dt_float*) calloc(hidden_max, sizeof(dt_float));
	}

	w_ho = (dt_float**) calloc(hidden_max, sizeof(dt_float*));
	for(j = 0; j < hidden_max; j++) {
		w_ho[j] = (dt_float*) calloc(output_max, sizeof(dt_float));
	}

	patterns = (dt_int*) calloc(pattern_max, sizeof(dt_int));
	error = (dt_float*) calloc(output_max, sizeof(dt_float));

#ifdef FLAG_NEGATIVE_SAMPLING
	samples = (dt_int**) calloc(pattern_max, sizeof(dt_int*));
	for(p = 0; p < pattern_max; p++) {
		samples[p] = (dt_int*) calloc(NEGATIVE_SAMPLES_MAX, sizeof(dt_int));
	}
#endif
}

static void resources_release() {
	for(i = 0; i < context_total_sentences; i++) {
		for(j = 0; j < context_total_words[i]; j++) {
			free(context[i][j]);
		}
		free(context[i]);
	}
	free(context);

	free(context_total_words);

	free(onehot);
	free(input);
	free(hidden);
	free(output);
	free(output_raw);

	for(p = 0; p < pattern_max; p++) {
		free(target[p]);
	}
	free(target);

	for(i = 0; i < input_max; i++) {
		free(w_ih[i]);
	}
	free(w_ih);

	for(j = 0; j < hidden_max; j++) {
		free(w_ho[j]);
	}
	free(w_ho);

	free(patterns);
	free(error);

#ifdef FLAG_NEGATIVE_SAMPLING
	for(p = 0; p < pattern_max; p++) {
		free(samples[p]);
	}
	free(samples);
#endif
}

static dt_int cmp_words(const void* a, const void* b) {
	dt_float diff = (*(xWord*) a).prob - (*(xWord*) b).prob;

	return diff < 0 ? 1 : diff > 0 ? -1 : 0;
}

static void initialize_vocab() {
	ffilter = fopen(FILTER_PATH, "r");

	if(!ffilter) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
#endif
		return;
	}

	parse_corpus();
	resources_allocate();

	dt_int index = 0;
	bst_to_map(vocab, &index);

	for(i = 0; i < context_total_sentences; i++) {
		for(j = 0; j < context_total_words[i]; j++) {
			if(!context[i][j] || !context[i][j][0]) {
				continue;
			}

			index = word_to_index(context[i][j]);

			if(!index_valid(index)) {
				continue;
			}

			xWord* center = index_to_word(index);

			for(k = j - WINDOW_MAX; k <= j + WINDOW_MAX; k++) {
				if(k == j || k < 0 || k >= context_total_words[i] || !context[i][k] || !context[i][k][0]) {
					continue;
				}

				dt_int context_index = word_to_index(context[i][k]);

				if(!index_valid(context_index)) {
					continue;
				}

				if(!contains_context(center, index, context_index)) {
					target[index][(center->context_count)++] = context_index;
				}
			}
		}
	}

#ifdef FLAG_NEGATIVE_SAMPLING
	bst_freq_sum(vocab, (vocab_freq_sum = 0, &vocab_freq_sum));
	bst_freq_max(vocab, (vocab_freq_max = 0, &vocab_freq_max));
#endif

	for(p = 0; p < pattern_max; p++) {
		patterns[p] = p;
	}
}

static void initialize_test() {
#ifdef FLAG_DEBUG
	printf("Center:\t\t%s\n\n", test_word);
#endif

	dt_int index = word_to_index(test_word);

	if(!index_valid(index)) {
		return;
	}

	onehot_set(input, index, input_max);

	xWord* center_word = index_to_word(index);

	for(k = 0; k < center_word->context_count; k++) {
#ifdef FLAG_DEBUG
		xWord* context = index_to_word(target[index][k]);
		printf("Context #%d:\t%s\n", k + 1, context->word);
#endif
	}

#ifdef FLAG_DEBUG
	printf("\n");
#endif
}

static void initialize_weights() {
	for(i = 0; i < input_max; i++) {
		for(j = 0; j < hidden_max; j++) {
			w_ih[i][j] = 2.0 * (random() - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}

	for(j = 0; j < hidden_max; j++) {
		for(k = 0; k < output_max; k++) {
			w_ho[j][k] = 2.0 * (random() - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}
}

static void initialize_epoch() {
	for(p = 0; p < pattern_max; p++) {
		p1 = p + random() * (pattern_max - p);

		p2 = patterns[p];
		patterns[p] = patterns[p1];
		patterns[p1] = p2;
	}

	alpha = max(LEARNING_RATE_MIN, LEARNING_RATE_MAX * (1 - (dt_float) epoch / EPOCH_MAX));

	loss = 0.0;
}

static void initialize_input() {
	onehot_set(input, patterns[p1], input_max);
}

static void forward_propagate_input_layer() {
	for(j = 0; j < hidden_max; j++) {
		hidden[j] = 0.0;

		for(i = 0; i < input_max; i++) {
			hidden[j] += input[i].on * w_ih[i][j];
		}
	}
}

static void forward_propagate_hidden_layer() {
	for(k = 0; k < output_max; k++) {
		output[k] = 0.0;

		for(j = 0; j < hidden_max; j++) {
			output[k] += hidden[j] * w_ho[j][k];
		}
	}
}

static void normalize_output_layer() {
	dt_float out_max = DT_FLOAT_MIN;

	for(k = 0; k < output_max; k++) {
		out_max = max(out_max, output_raw[k] = output[k]);
	}

	dt_float out_exp[output_max];
	sum = 0.0;

	for(k = 0; k < output_max; k++) {
		sum += out_exp[k] = exp(output[k] - out_max);
	}

	for(k = 0; k < output_max; k++) {
		output[k] = out_exp[k] / sum;
	}
}

static void calculate_error() {
	dt_int context_max = index_to_word(p)->context_count;
	dt_float error_t[context_max][output_max];

#ifdef FLAG_NEGATIVE_SAMPLING
	dt_int ck_max = 0;
	dt_int ck[context_max * NEGATIVE_SAMPLES_MAX];
	memset(ck, -1, context_max * NEGATIVE_SAMPLES_MAX * sizeof(dt_int));

	for(c = 0; c < context_max; c++) {
		for(k = 0; k < NEGATIVE_SAMPLES_MAX; k++) {
			if(samples[c][k] < 0) {
				continue;
			}

			dt_int smp = samples[c][k];
			ck[ck_max++] = smp;

			error_t[c][smp] = output[smp] - (smp == target[p][c]);
		}
	}

	for(k = 0; k < ck_max; k++) {
		error[ck[k]] = 0.0;

		for(c = 0; c < context_max; c++) {
			error[ck[k]] += error_t[c][ck[k]];
		}
	}

#else
	for(c = 0; c < context_max; c++) {
		for(k = 0; k < output_max; k++) {
			error_t[c][k] = output[k] - (k == target[p][c]);
		}
	}

	for(k = 0; k < output_max; k++) {
		error[k] = 0.0;

		for(c = 0; c < context_max; c++) {
			error[k] += error_t[c][k];
		}
	}
#endif
}

static void update_hidden_layer_weights() {
#ifdef FLAG_NEGATIVE_SAMPLING
	dt_int context_max = index_to_word(p)->context_count;
#endif

	for(j = 0; j < hidden_max; j++) {
#ifdef FLAG_NEGATIVE_SAMPLING
		for(c = 0; c < context_max; c++) {
			for(k = 0; k < NEGATIVE_SAMPLES_MAX; k++) {
				if(samples[c][k] < 0) {
					continue;
				}

				dt_int ck = samples[c][k];

				w_ho[j][ck] -= alpha * hidden[j] * error[ck];
			}
		}
#else
		for(k = 0; k < output_max; k++) {
			w_ho[j][k] -= alpha * hidden[j] * error[k];
		}
#endif
	}
}

static void update_input_layer_weights() {
	dt_float error_t[output_max];

#ifdef FLAG_NEGATIVE_SAMPLING
	dt_int context_max = index_to_word(p)->context_count;

	for(j = 0; j < hidden_max; j++) {
		error_t[j] = 0.0;

		for(c = 0; c < context_max; c++) {
			for(k = 0; k < NEGATIVE_SAMPLES_MAX; k++) {
				if(samples[c][k] < 0) {
					continue;
				}

				error_t[j] += error[samples[c][k]] * w_ho[j][samples[c][k]];
			}
		}
	}
#else
	for(j = 0; j < hidden_max; j++) {
		error_t[j] = 0.0;

		for(k = 0; k < output_max; k++) {
			error_t[j] += error[k] * w_ho[j][k];
		}
	}
#endif

	for(j = 0; j < hidden_max; j++) {
		w_ih[p][j] -= alpha * input[p].on * error_t[j];
	}
}

#ifdef FLAG_NEGATIVE_SAMPLING
static void negative_sampling() {
	dt_int exit;
	dt_float freq, rnd;

	dt_int context_max = index_to_word(p)->context_count;
	dt_float max_freq = ((dt_float) vocab_freq_max / vocab_freq_sum);

	for(c = 0; c < context_max; c++) {
		memset(samples[c], -1, NEGATIVE_SAMPLES_MAX * sizeof(dt_int));

		for(samples[c][0] = k = 0; k < output_max && k != target[p][c]; samples[c][0] = ++k)
			;

		for(k = 0; k < NEGATIVE_SAMPLES_MAX; k++) {
			if(k) {
				for(exit = 0; exit < MONTE_CARLO_EMERGENCY; exit++) {
					samples[c][k] = random_int() % pattern_max;
					freq = (dt_float) index_to_word(samples[c][k])->freq / vocab_freq_sum;

					rnd = random() * max_freq * MONTE_CARLO_FACTOR;

					if(samples[c][k] != samples[c][0] && rnd < freq) {
						break;
					}
				}
			}
		}
	}
}
#endif

static void calculate_loss() {
	dt_int context_max = index_to_word(p)->context_count;

	for(c = 0; c < context_max; c++) {
		loss -= output_raw[target[p][c]];
	}

	sum = 0.0;

	for(k = 0; k < output_max; k++) {
		sum += exp(output_raw[k]);
	}

	loss += context_max * log(sum);
}

void nn_start() {
	static dt_int done = 0;

	if(done++) {
		return;
	}

	srand(time(0));

#ifdef FLAG_LOG
	flog = fopen(LOG_PATH, "w");
#endif

	initialize_vocab();
}

void nn_finish() {
	static dt_int done = 0;

	if(done++) {
		return;
	}

#ifdef FLAG_DEBUG
#ifdef FLAG_PRINT_VOCAB
	screen_clear();
	dt_int index = 0;
	bst_print(vocab, &index);
#endif
#ifdef FLAG_PRINT_ERRORS
	screen_clear();
	invalid_index_print();
#endif
#endif

	bst_release(vocab);
	resources_release();

	if(fclose(ffilter) == EOF) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
#endif
	}

#ifdef FLAG_LOG
	if(fclose(flog) == EOF) {
		fprintf(flog, FILE_ERROR_MESSAGE);
	}
#endif
}

void training_run() {
	initialize_weights();

#ifdef FLAG_DEBUG
	clock_t start_time = clock();
#endif

	for(epoch = 0; epoch < EPOCH_MAX; epoch++) {
#ifdef FLAG_DEBUG
		screen_clear();
		printf("Epoch:\t%d / %d\n", epoch + 1, EPOCH_MAX);
#endif

		initialize_epoch();

		elapsed_time = clock();

		for(p1 = 0; p1 < pattern_max; p1++) {
			initialize_input();
			forward_propagate_input_layer();
			forward_propagate_hidden_layer();
			normalize_output_layer();
#ifdef FLAG_NEGATIVE_SAMPLING
			negative_sampling();
#endif
			calculate_error();
			update_hidden_layer_weights();
			update_input_layer_weights();
			calculate_loss();
		}

#ifdef FLAG_LOG
		fprintf(flog, "%cEpoch\t%d\n", epoch ? '\n' : '\0', epoch + 1);
		fprintf(flog, "Took\t%lf sec\n", time_get(elapsed_time));
		fprintf(flog, "Loss\t%lf\n", loss);
#endif
	}

#ifdef FLAG_DEBUG
	printf("Took:\t%lf sec\n", time_get(start_time));
#endif
}

void test_run(dt_char* word, dt_int count, dt_int* result) {
	test_word = word;

#ifdef FLAG_DEBUG
	screen_clear();
#endif
	initialize_test();
	forward_propagate_input_layer();
	forward_propagate_hidden_layer();
	normalize_output_layer();

	xWord pred[output_max];

	for(k = 0; k < output_max; k++) {
		dt_int index = k;
		pred[k] = *index_to_word(index);
		pred[k].prob = output[k];
	}

	qsort(pred, output_max, sizeof(xWord), cmp_words);

	xWord* center = index_to_word(p);

	dt_int index;

	for(index = 1, k = 0; k < count; k++, index++) {
		if(!strcmp(pred[k].word, word)) {
			count++;
			continue;
		}

		dt_int context_index = word_to_index(pred[k].word);

		if(!index_valid(context_index)) {
			continue;
		}

		if(index == 1) {
			*result = contains_context(center, p, context_index);
		}

#ifdef FLAG_DEBUG
		printf("#%d\t%lf\t%s\n", index, pred[k].prob, pred[k].word);
#endif
	}
}

void weights_save() {
	FILE* fwih = fopen(WEIGHTS_IH_PATH, "w");
	FILE* fwho = fopen(WEIGHTS_HO_PATH, "w");

	if(!fwih || !fwho) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
#endif
		return;
	}

	for(i = 0; i < input_max; i++) {
		for(j = 0; j < hidden_max; j++) {
			fprintf(fwih, "%s%lf", j ? " " : "", w_ih[i][j]);
		}

		fprintf(fwih, "\n");
	}

	for(j = 0; j < hidden_max; j++) {
		for(k = 0; k < output_max; k++) {
			fprintf(fwho, "%s%lf", k ? " " : "", w_ho[j][k]);
		}

		fprintf(fwho, "\n");
	}

	if(fclose(fwih) == EOF || fclose(fwho) == EOF) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
#endif
	}
}

void weights_load() {
	FILE* fwih = fopen(WEIGHTS_IH_PATH, "r");
	FILE* fwho = fopen(WEIGHTS_HO_PATH, "r");

	if(!fwih || !fwho) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
#endif
		return;
	}

	for(i = 0; i < input_max; i++) {
		for(j = 0; j < hidden_max; j++) {
			fscanf(fwih, "%lf", &w_ih[i][j]);
		}
	}

	for(j = 0; j < hidden_max; j++) {
		for(k = 0; k < output_max; k++) {
			fscanf(fwho, "%lf", &w_ho[j][k]);
		}
	}

	if(fclose(fwih) == EOF || fclose(fwho) == EOF) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
#endif
	}
}

void sentence_encode(dt_char* sentence, dt_float* vector) {
	memset(vector, 0, HIDDEN_MAX * sizeof(dt_float));

	dt_int index;
	dt_char* sep = WORD_DELIMITERS;
	dt_char* tok = strtok(sentence, sep);

	while(tok) {
		word_end(tok);

		if(!word_filter(tok)) {
			index = word_to_index(tok);

			if(!index_valid(index)) {
				continue;
			}

			for(j = 0; j < HIDDEN_MAX; j++) {
				vector[j] += w_ih[index][j];
			}
		}

		tok = strtok(NULL, sep);
	}
}
