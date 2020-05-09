#include "nn.h"

static clock_t elapsed_time;

static dt_int epoch;
static dt_float alpha;
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

static dt_int invalid_index[INVALID_INDEX_MAX];
static dt_int invalid_index_last;

#ifdef FLAG_NEGATIVE_SAMPLING
static dt_int ck, corpus_freq_sum, corpus_freq_max;
static xWord** samples;
#endif

#ifdef FLAG_LOG_FILE
static FILE* flog;
#endif

static void memcheck_log(void*, const dt_char*, const dt_char*, dt_int);

#ifdef FLAG_LOG
static dt_float time_get(clock_t start) {
	return (dt_float)(clock() - start) / CLOCKS_PER_SEC;
}

static void timestamp() {
	struct timeval tv;
	dt_char s[20];
	gettimeofday(&tv, NULL);
	strftime(s, sizeof(s) / sizeof(*s), "%d-%m-%Y %H:%M:%S", gmtime(&tv.tv_sec));
	fprintf(flog, "[%s.%03d] ", s, tv.tv_usec / 1000);
}

static void color_set(eColor color) {
	color == NONE ? fprintf(flog, "\033[0m") : fprintf(flog, "\033[1;3%dm", color);
}

static void echo_color(eColor color, const dt_char* format, ...) {
	va_list args;
	va_start(args, format);
	color_set(GRAY);
	timestamp();
	color_set(color);
	dt_char* f = (dt_char*) calloc(strlen(format) + 1, sizeof(dt_char));
	memcheck(f);
	strcpy(f, format);
	strcat(f, "\n");
	vfprintf(flog, f, args);
	color_set(NONE);
	va_end(args);
}
#endif

static void sigget(dt_int sig) {
	void* ptrs[BACKTRACE_DEPTH];
	size_t size = backtrace(ptrs, BACKTRACE_DEPTH);
	dt_char** stack = backtrace_symbols(ptrs, size);
	for(i = 0; i < size; i++) {
		stack[i][strchr(stack[i] + 4, ' ') - stack[i]] = '\0';
		echo_fail("%s @ %s", stack[i] + 4, stack[i] + 40);
	}
	free(stack);
	exit(1);
}

static void memcheck_log(void* ptr, const dt_char* file, const dt_char* func, dt_int line) {
	if(!ptr) {
#ifdef FLAG_LOG
		echo_fail(ERROR_MEMORY " @ %s:%s:%d", file, func, line);
#endif
		exit(1);
	}
}

#ifdef FLAG_NEGATIVE_SAMPLING
static dt_int cmp_freq(const void* a, const void* b) {
	dt_int diff = (*(xWord**) b)->freq - (*(xWord**) a)->freq;

	return diff < 0 ? 1 : diff > 0 ? -1 : 0;
}
#endif

static dt_int cmp_word(const void* a, const void* b) {
	dt_float diff = (*(xWord**) a)->prob - (*(xWord**) b)->prob;

	return diff < 0 ? 1 : diff > 0 ? -1 : 0;
}

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
		echo("Context #%d:\t%s", ++index, root->word->word);
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
		memcheck(root->target);
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
static void bst_extract(xWord* root, xWord** samples) {
	if(root) {
		*(samples + root->index) = root;
		bst_extract(root->left, samples);
		bst_extract(root->right, samples);
	}
}

static void bst_sample(xWord* root) {
	xWord** copies = (xWord**) calloc(pattern_max, sizeof(xWord*));
	memcheck(copies);
	bst_extract(root, copies);
	qsort(copies, pattern_max, sizeof(xWord*), cmp_freq);
	
	for(p = 0; p < pattern_max; p++) {
		ck = pattern_max / 2 + (p > 0) * p / 2 * (1 + 2 * (p % 2 - 1)) + p % 2;
		samples[ck] = copies[p];
	}
}

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
		echo_info("Corpus #%d:\t%s (%d)", root->index, root->word, root->freq);
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
	memcheck(node);
	node->word = (dt_char*) calloc(strlen(word), sizeof(dt_char));
	memcheck(node->word);
	strcpy(node->word, word);
	node->index = node->prob = node->context_max = node->freq = 0;
	node->left = node->right = node->next = NULL;
	node->context = NULL;
	node->target = NULL;
	return node;
}

static xContext* node_context_create(xWord* word) {
	xContext* node = (xContext*) calloc(1, sizeof(xContext));
	memcheck(node);
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

	if(root->word) {
		free(root->word);
		root->word = NULL;
	}

	root->left = root->right = root->next = NULL;
	root->index = root->prob = root->context_max = root->freq = 0;
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

#ifdef FLAG_PRINT_INDEX_ERRORS
static void invalid_index_print() {
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

static dt_int word_to_index(const dt_char* word) {
	dt_int index = *map_get(word);

	if(index < pattern_max) {
		return index;
	}

	echo_fail("%s not found in corpus", word);

	return -1;
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

#ifdef FLAG_STEM
	word[stem(word, 0, strlen(word) - 1) + 1] = '\0';
#endif
}

static dt_int word_stop(const dt_char* word) {
	if(strlen(word) < 3) {
		return 1;
	}

	const dt_char* p;
	for(p = word; *p && (isalpha(*p) || *p == '-'); p++)
		;

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
#ifdef FLAG_LOG
	echo("Allocating resources");
#endif

	onehot = (dt_int*) calloc(pattern_max, sizeof(dt_int));
	memcheck(onehot);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "onehot", 1, pattern_max);
#endif

	input = (xBit*) calloc(input_max, sizeof(xBit));
	memcheck(input);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "input", 1, input_max);
#endif

	hidden = (dt_float*) calloc(hidden_max, sizeof(dt_float));
	memcheck(hidden);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "hidden", 1, hidden_max);
#endif

	output = (dt_float*) calloc(output_max, sizeof(dt_float));
	memcheck(output);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "output", 1, output_max);
#endif

	output_raw = (dt_float*) calloc(output_max, sizeof(dt_float));
	memcheck(output_raw);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "output_raw", 1, output_max);
#endif

	w_ih = (dt_float**) calloc(input_max, sizeof(dt_float*));
	memcheck(w_ih);
	for(i = 0; i < input_max; i++) {
		w_ih[i] = (dt_float*) calloc(hidden_max, sizeof(dt_float));
		memcheck(w_ih[i]);
	}
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "w_ih", input_max, hidden_max);
#endif

	w_ho = (dt_float**) calloc(hidden_max, sizeof(dt_float*));
	memcheck(w_ho);
	for(j = 0; j < hidden_max; j++) {
		w_ho[j] = (dt_float*) calloc(output_max, sizeof(dt_float));
		memcheck(w_ho[j]);
	}
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "w_ho", hidden_max, output_max);
#endif

	patterns = (dt_int*) calloc(pattern_max, sizeof(dt_int));
	memcheck(patterns);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "patterns", 1, pattern_max);
#endif

	error = (dt_float*) calloc(output_max, sizeof(dt_float));
	memcheck(error);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "error", 1, output_max);
#endif

#ifdef FLAG_NEGATIVE_SAMPLING
	samples = (xWord**) calloc(pattern_max, sizeof(xWord*));
	memcheck(samples);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "samples", 1, pattern_max);
#endif
#endif

#ifdef FLAG_LOG
	echo_succ("Done allocating resources");
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
	free(samples);
	samples = NULL;
#endif
}

static void initialize_corpus() {
#ifdef FLAG_LOG
	echo("Initializing corpus");
	elapsed_time = clock();
#endif

	FILE* fin = fopen(CORPUS_PATH, "r");
	FILE* fstop = fopen(STOP_PATH, "r");

	if(!fstop || !fstop) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
		return;
	}

#ifdef FLAG_LOG
	echo("Loading stop words");
#endif

	dt_char line[LINE_CHARACTER_MAX];
	xWord* node;

	while(fgets(line, LINE_CHARACTER_MAX, fstop)) {
		line[strlen(line) - 1] = '\0';
		node = node_create(line);
		stops = list_insert(stops, &node);
	}

#ifdef FLAG_LOG
	echo_succ("Done loading stop words");
#endif

	dt_int success, sent_end;
	const dt_char* sep = WORD_DELIMITERS;
	dt_char* tok;
	xWord* window[WINDOW_MAX] = { 0 };

#ifdef FLAG_LOG
	echo("Reading corpus file");
#endif

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

#ifdef FLAG_LOG
					if(!(pattern_max % LOG_PERIOD_CORPUS)) {
						echo("Corpus grown to %d", pattern_max);
					}
#endif
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

#ifdef FLAG_LOG
	echo_succ("Done reading corpus file");
	echo_info("Corpus size: %d words", pattern_max);
#endif

	resources_allocate();

	for(p = 0; p < pattern_max; p++) {
		patterns[p] = p;
	}

#ifdef FLAG_LOG
	echo("Creating corpus map");
#endif

	dt_int index = 0;
	bst_to_map(corpus, &index);

#ifdef FLAG_LOG
	echo_succ("Done creating corpus map");
#endif

#ifdef FLAG_PRINT_CORPUS
	bst_print(corpus);
#endif

#ifdef FLAG_LOG
	echo("Building word targets");
#endif

	bst_target(corpus);

#ifdef FLAG_LOG
	echo_succ("Done building word targets");
#endif

#ifdef FLAG_NEGATIVE_SAMPLING
#ifdef FLAG_LOG
	echo("Calculating word frequency");
#endif	
	bst_sample(corpus);
	bst_freq_sum(corpus, (corpus_freq_sum = 0, &corpus_freq_sum));
	bst_freq_max(corpus, (corpus_freq_max = 0, &corpus_freq_max));
#endif

	if(fclose(fin) == EOF || fclose(fstop) == EOF) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
	}

#ifdef FLAG_LOG
	echo_succ("Done initializing corpus (%lf sec)", time_get(elapsed_time));
#endif
}

static void initialize_weights() {
#ifdef FLAG_LOG
	echo("Initializing weights");
#endif

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

#ifdef FLAG_LOG
	echo_succ("Done initializing weights");
#endif
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
		hidden[j] = w_ih[p][j];
	}
}

static void forward_propagate_hidden_layer() {
	for(k = 0; k < output_max; k++) {
#ifdef FLAG_DROPOUT
		if(random(0, 1) < DROPOUT_RATE_MAX) {
			continue;
		}
#endif

		for(output[k] = j = 0; j < hidden_max; j++) {
			output[k] += hidden[j] * w_ho[j][k];
		}
	}
}

#ifdef FLAG_NEGATIVE_SAMPLING
static void negative_sampling() {
	dt_float e, delta_ih[hidden_max], delta_ho;

#ifdef FLAG_MONTE_CARLO
	dt_int exit;
	dt_float freq, rnd;
	dt_float max_freq = ((dt_float) corpus_freq_max / corpus_freq_sum);
#endif

	for(c = 0; c < center->context_max; c++) {
		memset(delta_ih, 0, hidden_max * sizeof(dt_float));

		for(e = ck = 0; ck < NEGATIVE_SAMPLES_MAX; ck++) {
			if(ck) {
#ifdef FLAG_MONTE_CARLO
				for(exit = 0; exit < MONTE_CARLO_EMERGENCY; exit++) {
					k = (dt_int) random(0, pattern_max);
					freq = (dt_float) index_to_word(k)->freq / corpus_freq_sum;
					rnd = random(0, 1) * max_freq * MONTE_CARLO_FACTOR;

					if(k != center->target[c] && rnd > freq) {
						break;
					}
				}
#else
				k = (dt_int) random(0, pattern_max);
#endif
			}

			for(j = 0; j < hidden_max; j++) {
				e += w_ih[p][j] * w_ho[j][k];
			}

			delta_ho = alpha * (!ck - 1.0 / (1.0 + exp(e)));

			for(j = 0; j < hidden_max; j++) {
				delta_ih[j] += delta_ho * w_ho[j][k];
				w_ho[j][k] += delta_ho * w_ih[p][j];
			}
		}

		for(j = 0; j < hidden_max; j++) {
			w_ih[p][j] += delta_ih[j];
		}
	}
}
#else
static void normalize_output_layer() {
	dt_float out_max = DT_FLOAT_MIN;

	for(k = 0; k < output_max; k++) {
		out_max = max(out_max, output_raw[k] = output[k]);
	}

	dt_float out_exp[output_max];
	dt_float sum = 0.0;

	for(k = 0; k < output_max; k++) {
		sum += out_exp[k] = exp(output[k] - out_max);
	}

	for(k = 0; k < output_max; k++) {
		output[k] = out_exp[k] / sum;
	}
}

static void calculate_error() {
	for(k = 0; k < output_max; k++) {
		for(error[k] = c = 0; c < center->context_max; c++) {
			error[k] += output[k] - (k == center->target[c]->index);
		}
	}
}

static void update_hidden_layer_weights() {
	for(j = 0; j < hidden_max; j++) {
		for(k = 0; k < output_max; k++) {
			w_ho[j][k] -= alpha * error[k] * hidden[j];
		}
	}
}

static void update_input_layer_weights() {
	dt_float l;

	for(j = 0; j < hidden_max; j++) {
		for(l = k = 0; k < output_max; k++) {
			l += error[k] * w_ho[j][k];
		}

		w_ih[p][j] -= alpha * l;
	}
}
#endif

static void test_predict(const dt_char* word, dt_int count, dt_int* success) {
#ifdef FLAG_LOG
	echo_info("Center: %s", word);
	elapsed_time = clock();
#endif

	dt_int index = word_to_index(word);

	if(!index_valid(index)) {
		return;
	}

	onehot_set(input, index, input_max);

	forward_propagate_input_layer();
	forward_propagate_hidden_layer();

#ifndef FLAG_NEGATIVE_SAMPLING
	normalize_output_layer();
#endif

	xWord* pred[pattern_max];

	for(k = 0; k < output_max; k++) {
		dt_int index = k;
		pred[k] = index_to_word(index);
		pred[k]->prob = output[k];
	}

	qsort(pred, pattern_max, sizeof(xWord*), cmp_word);

	xWord* center = index_to_word(p);

	for(index = 1, k = 0; k < count; k++) {
		if(!strcmp(pred[k]->word, word)) {
			count++;
			continue;
		}

		dt_int context_index = word_to_index(pred[k]->word);

		if(!index_valid(context_index)) {
			continue;
		}

		if(index == 1) {
			*success = 0;

			for(c = 0; c < center->context_max; c++) {
				if(center->target[c]->index == context_index) {
					*success = 1;
					break;
				}
			}
		}

#ifdef FLAG_LOG
		echo("#%d\t%lf\t%s", index++, pred[k]->prob, pred[k]->word);
#endif
	}

#ifdef FLAG_LOG
	echo_cond(*success, "Prediction %scorrect (%lf sec)", *success ? "" : "not ", time_get(elapsed_time));
#endif
}

void nn_start() {
	static dt_int done = 0;

	if(done++) {
		return;
	}

	signal(SIGBUS, sigget);
	signal(SIGFPE, sigget);
	signal(SIGILL, sigget);
	signal(SIGSEGV, sigget);

	srand(time(0));

#ifdef FLAG_LOG_FILE
	flog = fopen(LOG_PATH, "w");
#endif

	initialize_corpus();
	initialize_weights();
}

void nn_finish() {
	static dt_int done = 0;

	if(done++) {
		return;
	}

#ifdef FLAG_PRINT_INDEX_ERRORS
	invalid_index_print();
#endif

	bst_release(corpus);
	corpus = NULL;

	list_release(stops);
	stops = NULL;

	resources_release();

#ifdef FLAG_LOG_FILE
	if(fclose(flog) == EOF) {
		echo_fail(ERROR_FILE);
	}
#endif
}

void training_run() {
#ifdef FLAG_LOG
	clock_t start_time = clock();
	echo("Started training");
#endif

	for(epoch = 0; epoch < EPOCH_MAX; epoch++) {
#ifdef FLAG_LOG
		echo("Started epoch %d/%d", epoch + 1, EPOCH_MAX);
		elapsed_time = clock();
#endif

		initialize_epoch();

		for(p1 = 0; p1 < pattern_max; p1++) {
#ifdef FLAG_LOG
			if(!(p1 % LOG_PERIOD_PASS)) {
				echo("Started pass %d/%d", p1 + 1, pattern_max);
			}
#endif
			initialize_input();
#ifdef FLAG_NEGATIVE_SAMPLING
			negative_sampling();
#else
			forward_propagate_input_layer();
			forward_propagate_hidden_layer();
			normalize_output_layer();
			calculate_error();
			update_hidden_layer_weights();
			update_input_layer_weights();
#endif
		}

#ifdef FLAG_BACKUP_WEIGHTS
		weights_save();
#endif

#ifdef FLAG_LOG
		echo_succ("Finished epoch %d/%d (%lf sec)", epoch + 1, EPOCH_MAX, time_get(elapsed_time));
#endif
	}

#ifdef FLAG_LOG
	echo_succ("Finished training (%lf sec)", time_get(start_time));
#endif
}

void testing_run() {
#ifdef FLAG_LOG
	clock_t start_time = clock();
	echo("Started testing");
#endif

	FILE* ftest = fopen(TEST_PATH, "r");

	if(!ftest) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
		return;
	}

	dt_char line[LINE_CHARACTER_MAX];
	dt_int test_count = 0, success = 0, tries_sum = 0, sent_end;

	while(fgets(line, LINE_CHARACTER_MAX, ftest)) {
		line[strlen(line) - 1] = '\0';
		word_clean(line, &sent_end);

		if(!word_stop(line)) {
			test_predict(line, WINDOW_MAX, &success);
			test_count++, tries_sum += success;
		}
	}

	if(fclose(ftest) == EOF) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
	}

#ifdef FLAG_LOG
	dt_float prec = 100.0 * tries_sum / test_count;
	echo_cond(prec > 50.0, "Precision: %.1lf%%", prec);
	echo_succ("Finished testing (%lf sec)", time_get(start_time));
#endif
}

void weights_save() {
#ifdef FLAG_LOG
	echo("Started saving weights");
#endif

	FILE* fwih = fopen(WEIGHTS_IH_PATH, "w");
	FILE* fwho = fopen(WEIGHTS_HO_PATH, "w");

	if(!fwih || !fwho) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
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
		echo_fail(ERROR_FILE);
#endif
	}

#ifdef FLAG_LOG
	echo_succ("Finished saving weights");
#endif
}

void weights_load() {
#ifdef FLAG_LOG
	echo("Started loading weights");
#endif

	FILE* fwih = fopen(WEIGHTS_IH_PATH, "r");
	FILE* fwho = fopen(WEIGHTS_HO_PATH, "r");

	if(!fwih || !fwho) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
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
		echo_fail(ERROR_FILE);
#endif
	}

#ifdef FLAG_LOG
	echo_succ("Finished loading weights");
#endif
}

void sentence_encode(dt_char* sentence, dt_float* vector) {
#ifdef FLAG_LOG
	echo("Encoding sentence \"%s\"", sentence);
#endif

	memset(vector, 0, hidden_max * sizeof(dt_float));

	dt_int index, sent_end;
	const dt_char* sep = WORD_DELIMITERS;
	dt_char* tok = strtok(sentence, sep);

	while(tok) {
		word_clean(tok, &sent_end);

		if(!word_stop(tok)) {
			index = word_to_index(tok);

			if(!index_valid(index)) {
				continue;
			}

			for(j = 0; j < hidden_max; j++) {
				vector[j] += w_ih[index][j];
			}
		}

		tok = strtok(NULL, sep);
	}
}
