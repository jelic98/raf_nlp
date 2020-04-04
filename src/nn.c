#include "include/nn.h"

static clock_t elapsed_time;

static dt_int epoch;
static dt_float alpha, sum;
static xWord* center;

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
static dt_int* patterns;

static xWord* corpus;
static xWord* stops;
static dt_int* onehot;
static dt_char* test_word;

static dt_int invalid_index[INVALID_INDEX_MAX];
static dt_int invalid_index_last;

#ifdef FLAG_NEGATIVE_SAMPLING
static dt_int corpus_freq_sum, corpus_freq_max;
static dt_int** samples;
#endif

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

static xWord* node_create(const dt_char*);
static xContext* node_context_create(xWord*);
static void node_release(xWord*);
static void node_context_release(xContext*);

static xWord* list_insert(xWord* root, xWord** node) {
	if(root) {
		xWord* tmp = root;

		while(tmp->next) {
			if(!strcmp(tmp->word, (*node)->word)) {
				node_release(*node);
				*node = tmp;
				return root;
			}

			tmp = tmp->next;
		}

		tmp->next = *node;

		return root;
	} else {
		return *node;
	}

	return *node;
}

static xContext* list_context_insert(xContext* root, xWord* word, dt_int* success) {
	*success = 1;
	xContext* node = node_context_create(word);

	if(root) {
		xContext* tmp = root;

		while(tmp->next) {
			if(tmp->word == node->word) {
				node_context_release(node);
				*success = 0;
				return root;
			}

			tmp = tmp->next;
		}

		tmp->next = node;

		return root;
	} else {
		return node;
	}

	return node;
}

static dt_int list_contains(xWord* root, const dt_char* word) {
	while(root) {
		if(!strcmp(root->word, word)) {
			return 1;
		}

		root = root->next;
	}

	return 0;
}

#ifdef FLAG_PRINT_CORPUS
static void list_context_print(xContext* root) {
	dt_int index = 0;

	while(root) {
#ifdef FLAG_LOG
		fprintf(flog, "Context #%d:\t%s\n", ++index, root->word->word);
#endif
		root = root->next;
	}
}
#endif

static void list_release(xWord* root) {
	xWord* node;

	while(root) {
		node = root;
		root = root->next;
		node_release(node);
	}
}

static void list_context_release(xContext* root) {
	xContext* node;

	while(root) {
		node = root;
		root = root->next;
		node_context_release(node);
	}
}

static xWord* bst_insert(xWord* root, xWord** node, dt_int* success) {
	*success = 0;

	if(root) {
		dt_int cmp = strcmp(root->word, (*node)->word);

		if(cmp < 0) {
			root->left = bst_insert(root->left, node, success);
		} else if(cmp > 0) {
			root->right = bst_insert(root->right, node, success);
		} else {
			node_release(*node);
			*node = root;
			root->freq++;
		}

		return root;
	}

	(*node)->freq++;
	*success = 1;
	return *node;
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

static void bst_to_map(xWord* root, dt_int* index) {
	if(root) {
		bst_to_map(root->left, index);
		*map_get(root->word) = root->index = (*index)++;
		bst_to_map(root->right, index);
	}
}

static void bst_target(xWord* root) {
	if(root) {
		dt_int index = 0;
		root->target = (xWord**) calloc(root->context_max, sizeof(xWord*));
		xContext* tmp = root->context;

		while(tmp) {
			root->target[index++] = tmp->word;
			tmp = tmp->next;
		}

		list_context_release(root->context);
		root->context = NULL;

		bst_target(root->left);
		bst_target(root->right);
	}
}

#ifdef FLAG_NEGATIVE_SAMPLING
static void bst_freq_sum(xWord* root, dt_int* sum) {
	if(root) {
		*sum += root->freq;
		bst_freq_sum(root->left, sum);
		bst_freq_sum(root->right, sum);
	}
}

static void bst_freq_max(xWord* root, dt_int* max) {
	if(root) {
		*max = max(*max, root->freq);
		bst_freq_max(root->left, max);
		bst_freq_max(root->right, max);
	}
}
#endif

#ifdef FLAG_PRINT_CORPUS
static void bst_print(xWord* root) {
	if(root) {
		bst_print(root->left);
#ifdef FLAG_LOG
		fprintf(flog, "Corpus #%d:\t%s (%d)\n", root->index, root->word, root->freq);
		list_context_print(root->context);
#endif
		bst_print(root->right);
	}
}
#endif

static void bst_release(xWord* root) {
	if(root) {
		bst_release(root->left);
		bst_release(root->right);
		node_release(root);
	}
}

static xWord* node_create(const dt_char* word) {
	xWord* node = (xWord*) calloc(1, sizeof(xWord));
	node->word = (dt_char*) calloc(strlen(word), sizeof(dt_char));
	strcpy(node->word, word);
	node->index = node->prob = node->context_max = node->freq = 0;
	node->left = node->right = node->next = NULL;
	node->context = NULL;
	node->target = NULL;
	return node;
}

static xContext* node_context_create(xWord* word) {
	xContext* node = (xContext*) calloc(1, sizeof(xContext));
	node->word = word;
	return node;
}

static void node_release(xWord* root) {
	if(root->target) {
		free(root->target);
		root->target = NULL;
	}

	if(root->context) {
		list_context_release(root->context);
		root->context = NULL;
	}

	root->left = root->right = root->next = NULL;
	root->index = root->prob = root->context_max = root->freq = 0;
	free(root->word);
	root->word = NULL;
	free(root);
}

static void node_context_release(xContext* root) {
	root->word = NULL;
	free(root);
}

static xWord* index_to_word(dt_int index) {
	return bst_get(corpus, &index);
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

static void word_clean(dt_char* word, dt_int* sent_end) {
	word_lower(word);

	dt_int end = strlen(word) - 1;

	*sent_end = strchr(SENTENCE_DELIMITERS, word[end]) != NULL;

	if(*sent_end) {
		word[end] = '\0';
	}
}

static dt_int word_stop(dt_char* word) {
	if(strlen(word) < 3) {
		return 1;
	}

	dt_char* p;
	for(p = word; *p && (isalpha(*p) || *p == '-'); p++);

	return *p || list_contains(stops, word);
}

static void onehot_set(xBit* onehot, dt_int index, dt_int size) {
	dt_int q;

	for(q = 0; q < size; q++) {
		onehot[q].on = 0;
	}

	onehot[p = index].on = 1;
	center = index_to_word(p);
}

static void resources_allocate() {
	onehot = (dt_int*) calloc(pattern_max, sizeof(dt_int));
	input = (xBit*) calloc(input_max, sizeof(xBit));
	hidden = (dt_float*) calloc(hidden_max, sizeof(dt_float));
	output = (dt_float*) calloc(output_max, sizeof(dt_float));
	output_raw = (dt_float*) calloc(output_max, sizeof(dt_float));

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
	free(onehot);
	onehot = NULL;
	
	free(input);
	input = NULL;
	
	free(hidden);
	hidden = NULL;
	
	free(output);
	output = NULL;

	free(output_raw);
	output_raw = NULL;

	for(i = 0; i < input_max; i++) {
		free(w_ih[i]);
	}
	free(w_ih);
	w_ih = NULL;

	for(j = 0; j < hidden_max; j++) {
		free(w_ho[j]);
	}
	free(w_ho);
	w_ho = NULL;

	free(patterns);
	patterns = NULL;

	free(error);
	error = NULL;

#ifdef FLAG_NEGATIVE_SAMPLING
	for(p = 0; p < pattern_max; p++) {
		free(samples[p]);
	}
	free(samples);
	samples = NULL;
#endif
}

static void initialize_corpus() {
	FILE* fin = fopen(CORPUS_PATH, "r");
	FILE* fstop = fopen(STOP_PATH, "r");

	if(!fstop || !fstop) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
#endif
		return;
	}

	dt_char line[LINE_CHARACTER_MAX];
	xWord* node;

	while(fgets(line, LINE_CHARACTER_MAX, fstop)) {
		line[strlen(line) - 1] = '\0';
		node = node_create(line);
		stops = list_insert(stops, &node);
	}

	dt_int success, sent_end;
	dt_char* sep = WORD_DELIMITERS;
	dt_char* tok;
	xWord* window[WINDOW_MAX] = { 0 };

	while(fgets(line, LINE_CHARACTER_MAX, fin)) {
		tok = strtok(line, sep);

		while(tok) {
			word_clean(tok, &sent_end);

			if(!word_stop(tok)) {
				node = node_create(tok);
				corpus = bst_insert(corpus, &node, &success);

				if(success) {
					pattern_max = input_max = ++output_max;
					hidden_max = HIDDEN_MAX;
				}

				window[WINDOW_MAX - 1] = node;

				for(c = 0; c < WINDOW_MAX - 1; c++) {
					if(window[c] && strcmp(window[c]->word, node->word)) {
						node->context = list_context_insert(node->context, window[c], &success);
						node->context_max += success;

						window[c]->context = list_context_insert(window[c]->context, node, &success);
						window[c]->context_max += success;
					}

					window[c] = window[c + 1];
				}
			}

			if(sent_end) {
				memset(window, 0, WINDOW_MAX * sizeof(xWord*));
			}

			tok = strtok(NULL, sep);
		}
	}

	resources_allocate();

	for(p = 0; p < pattern_max; p++) {
		patterns[p] = p;
	}

	dt_int index = 0;
	bst_to_map(corpus, &index);

#ifdef FLAG_DEBUG
#ifdef FLAG_PRINT_CORPUS
	bst_print(corpus);
#endif
#endif

	bst_target(corpus);

#ifdef FLAG_NEGATIVE_SAMPLING
	bst_freq_sum(corpus, (corpus_freq_sum = 0, &corpus_freq_sum));
	bst_freq_max(corpus, (corpus_freq_max = 0, &corpus_freq_max));
#endif

	if(fclose(fin) == EOF || fclose(fstop) == EOF) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
#endif
	}
}

static dt_int cmp_words(const void* a, const void* b) {
	dt_float diff = (*(xWord*) a).prob - (*(xWord*) b).prob;

	return diff < 0 ? 1 : diff > 0 ? -1 : 0;
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

	for(c = 0; c < center_word->context_max; c++) {
#ifdef FLAG_DEBUG
		xWord* context = index_to_word(center->target[c]->index);
		printf("Context #%d:\t%s\n", c + 1, context->word);
#endif
	}

#ifdef FLAG_DEBUG
	printf("\n");
#endif
}

static void initialize_weights() {
	for(i = 0; i < input_max; i++) {
		for(j = 0; j < hidden_max; j++) {
			w_ih[i][j] = random(0, INITIAL_WEIGHT_MAX);
		}
	}

	for(j = 0; j < hidden_max; j++) {
		for(k = 0; k < output_max; k++) {
			w_ho[j][k] = random(0, INITIAL_WEIGHT_MAX);
		}
	}
}

static void initialize_epoch() {
	for(p = 0; p < pattern_max; p++) {
		p2 = patterns[p];
		patterns[p] = patterns[p1 = p + random(0, 1) * (pattern_max - p)];
		patterns[p1] = p2;
	}

	alpha = max(LEARNING_RATE_MIN, LEARNING_RATE_MAX * (1 - (dt_float) epoch / EPOCH_MAX));
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
	dt_float error_t[center->context_max][output_max];

	for(c = 0; c < center->context_max; c++) {
		for(k = 0; k < output_max; k++) {
			error_t[c][k] = output[k] - (k == center->target[c]->index);
		}
	}

	for(k = 0; k < output_max; k++) {
		error[k] = 0.0;

		for(c = 0; c < center->context_max; c++) {
			error[k] += error_t[c][k];
		}
	}
}

static void update_hidden_layer_weights() {
	for(j = 0; j < hidden_max; j++) {
		for(k = 0; k < output_max; k++) {
			w_ho[j][k] -= alpha * hidden[j] * error[k];
		}
	}
}

static void update_input_layer_weights() {
	dt_float error_t[hidden_max];

	for(j = 0; j < hidden_max; j++) {
		error_t[j] = 0.0;

		for(k = 0; k < output_max; k++) {
			error_t[j] += error[k] * w_ho[j][k];
		}
	}

	for(j = 0; j < hidden_max; j++) {
		w_ih[p][j] -= alpha * input[p].on * error_t[j];
	}
}

#ifdef FLAG_NEGATIVE_SAMPLING
static void negative_sampling() {
	dt_int exit;
	dt_float freq, rnd;

	dt_float max_freq = ((dt_float) corpus_freq_max / corpus_freq_sum);

	for(c = 0; c < center->context_max; c++) {
		memset(samples[c], -1, NEGATIVE_SAMPLES_MAX * sizeof(dt_int));

		for(samples[c][0] = k = 0; k < output_max && k != center->target[c]; samples[c][0] = ++k)
			;

		for(k = 0; k < NEGATIVE_SAMPLES_MAX; k++) {
			if(k) {
				for(exit = 0; exit < MONTE_CARLO_EMERGENCY; exit++) {
					samples[c][k] = random(0, pattern_max);
					freq = (dt_float) index_to_word(samples[c][k])->freq / corpus_freq_sum;

					rnd = random(0, 1) * max_freq * MONTE_CARLO_FACTOR;

					if(samples[c][k] != samples[c][0] && rnd < freq) {
						break;
					}
				}
			}
		}
	}
}
#endif

static void test_predict(dt_char* word, dt_int count, dt_int* result) {
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
			*result = 0;

			for(c = 0; c < center->context_max; c++) {
				if(center->target[c]->index == context_index) {
					*result = 1;
					break;
				}
			}
		}

#ifdef FLAG_DEBUG
		printf("#%d\t%lf\t%s\n", index, pred[k].prob, pred[k].word);
#endif
	}
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

	initialize_corpus();
}

void nn_finish() {
	static dt_int done = 0;

	if(done++) {
		return;
	}

#ifdef FLAG_DEBUG
#ifdef FLAG_PRINT_ERRORS
	screen_clear();
	invalid_index_print();
#endif
#endif

	bst_release(corpus);
	corpus = NULL;

	list_release(stops);
	stops = NULL;
	
	resources_release();

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
		}

#ifdef FLAG_LOG
		fprintf(flog, "%cEpoch\t%d\n", epoch ? '\n' : '\0', epoch + 1);
		fprintf(flog, "Took\t%lf sec\n", time_get(elapsed_time));
#endif
	}

#ifdef FLAG_DEBUG
	printf("Took:\t%lf sec\n", time_get(start_time));
#endif
}

void test_run() {
	FILE* ftest = fopen(TEST_PATH, "r");

	if(!ftest) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
#endif
		return;
	}

	dt_char line[LINE_CHARACTER_MAX];
	dt_int test_count = -1, result = 0, tries_sum = 0;

	while(test_count++, fgets(line, LINE_CHARACTER_MAX, ftest)) {
		line[strlen(line) - 1] = '\0';
		test_predict(line, 5, &result);
		tries_sum += result;
	}

	printf("\nPrecision: %.1lf%%\n", 100.0 * tries_sum / test_count);

	if(fclose(ftest) == EOF) {
#ifdef FLAG_LOG
		fprintf(flog, FILE_ERROR_MESSAGE);
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

	dt_int index, sent_end;
	dt_char* sep = WORD_DELIMITERS;
	dt_char* tok = strtok(sentence, sep);

	while(tok) {
		word_clean(tok, &sent_end);

		if(!word_stop(tok)) {
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
