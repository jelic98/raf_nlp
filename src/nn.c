#include "nn.h"

static struct timespec time_start;

static dt_float alpha;

#ifdef FLAG_CALCULATE_LOSS
static dt_float loss;
#endif

static dt_int pattern_max, input_max, hidden_max, output_max, token_max;

static dt_int* patterns;
static dt_float** w_ih;
static dt_float** w_ho;

#ifdef FLAG_NEGATIVE_SAMPLING
static dt_float* output;
#else
static dt_float** error;
static dt_float** output;
#endif

static xWord* stops;
static xWord** vocab;
static dt_int* vocab_hash;

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
static xWord** filter;
static dt_int filter_max;
#endif

static dt_int invalid_index[INVALID_INDEX_MAX];
static dt_int invalid_index_last;

#ifdef FLAG_NEGATIVE_SAMPLING
static dt_int corpus_freq_sum, corpus_freq_max;
#ifndef FLAG_MONTE_CARLO
static xWord** samples;
#endif
#endif

static xThread thread_args[THREAD_MAX];
static pthread_t thread_ids[THREAD_MAX];
static pthread_mutex_t mtx_count_epoch;
static sem_t* sem_epoch_1;
static sem_t* sem_epoch_2;
static dt_int count_epoch;

#ifdef FLAG_LOG_FILE
static FILE* flog;
#endif

#ifdef FLAG_LOG
static dt_int time_get(struct timespec start) {
	struct timespec finish;
	clock_gettime(CLOCK_MONOTONIC, &finish);
	return finish.tv_sec - start.tv_sec;
}

static void timestamp() {
	struct timeval tv;
	dt_char s[20];
	gettimeofday(&tv, NULL);
	strftime(s, sizeof(s) / sizeof(*s), "%d-%m-%Y %H:%M:%S", gmtime(&tv.tv_sec));
	fprintf(flog, "[%s.%03d] ", s, (dt_int) tv.tv_usec / 1000);
}

#ifdef FLAG_COLOR_LOG
static void color_set(eColor color) {
	color == NONE ? fprintf(flog, "\033[0m") : fprintf(flog, "\033[1;3%dm", color);
}
#endif

static void echo_color(eColor color, dt_int replace, const dt_char* format, ...) {
	va_list args;
	va_start(args, format);
#ifdef FLAG_COLOR_LOG
	color_set(GRAY);
#endif
	if(!replace) {
		timestamp();
	}
#ifdef FLAG_COLOR_LOG
	color_set(color);
#endif
	dt_char* f = (dt_char*) calloc(strlen(format) + 1, sizeof(dt_char));
	if(replace) {
		strcat(f, "\r");
		strcat(f, format);
	} else {
		strcpy(f, format);
		strcat(f, "\n");
	}
	vfprintf(flog, f, args);
#ifdef FLAG_COLOR_LOG
	color_set(NONE);
#endif
	va_end(args);
}
#endif

static void sigget(dt_int sig) {
	void* ptrs[BACKTRACE_DEPTH];
	size_t size = backtrace(ptrs, BACKTRACE_DEPTH);
	dt_char** stack = backtrace_symbols(ptrs, size);
	dt_int i;
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

static void vector_normalize(dt_float* vector, dt_int size) {
	dt_float sum;
	dt_int q;

	for(sum = q = 0; q < size; q++) {
		sum += vector[q] * vector[q];
	}

	for(sum = sqrt(sum), q = 0; q < size; q++) {
		vector[q] /= sum;
	}
}

#if defined(FLAG_TEST_CONTEXT) || !defined(FLAG_NEGATIVE_SAMPLING)
static void vector_softmax(dt_float* vector, dt_int size) {
	dt_int k;
	dt_float sum, vector_exp[size];

	for(sum = k = 0; k < size; k++) {
		sum += vector_exp[k] = exp(vector[k]);
	}

	for(k = 0; k < output_max; k++) {
		vector[k] = vector_exp[k] / sum;
	}
}
#endif

#ifdef FLAG_TEST_SIMILARITY
static void vector_distance(dt_float* v1, dt_float* v2, dt_int size, dt_float* dist) {
	dt_int k;
	dt_float sum;

#ifdef FLAT_DISTANCE_COSINE
	dt_float sum1, sum2;

	for(sum = sum1 = sum2 = k = 0; k < size; k++) {
		sum += v1[k] * v2[k];
		sum1 += v1[k] * v1[k];
		sum2 += v2[k] * v2[k];
	}

	sum1 = sqrt(sum1);
	sum2 = sqrt(sum2);

	*dist = sum / (sum1 * sum2);
#else
	dt_float diff;

	for(sum = k = 0; k < size; k++) {
		diff = v1[k] - v2[k];
		sum += diff * diff;
	}

	*dist = sqrt(sum);
#endif
}
#endif

static dt_int cmp_int(const void* a, const void* b) {
	return *(dt_int*) a - *(dt_int*) b;
}

#ifdef FLAG_FILTER_VOCABULARY_HIGH
static dt_int cmp_freq(const void* a, const void* b) {
	dt_int diff = (*(xWord**) b)->freq - (*(xWord**) a)->freq;

	return diff < 0 ? 1 : diff > 0 ? -1 : 0;
}
#endif

#ifdef FLAG_NEGATIVE_SAMPLING
#ifndef FLAG_MONTE_CARLO
#ifndef FLAG_UNIGRAM_DISTRIBUTION
static dt_int cmp_freq_dist(const void* a, const void* b) {
	dt_float diff = (*(xWord**) b)->freq_dist - (*(xWord**) a)->freq_dist;

	return diff < 0 ? 1 : diff > 0 ? -1 : 0;
}
#endif
#endif
#endif

#ifdef FLAG_TEST_SIMILARITY
static dt_int cmp_dist(const void* a, const void* b) {
	dt_float diff = (*(xWord**) b)->dist - (*(xWord**) a)->dist;

	return diff < 0 ? 1 : diff > 0 ? -1 : 0;
}
#endif

#ifdef FLAG_TEST_CONTEXT
static dt_int cmp_prob(const void* a, const void* b) {
	dt_float diff = (*(xWord**) a)->prob - (*(xWord**) b)->prob;

	return diff < 0 ? 1 : diff > 0 ? -1 : 0;
}
#endif

static dt_uint hash_get(const dt_char* word) {
	dt_ull i, h;

	for(h = i = 0; word[i]; i++) {
		h = h * 257 + word[i];
	}

	return h % VOCABULARY_HASH_MAX;
}

static void map_init(xWord** vocab) {
	dt_uint p, h;
		
	for(h = 0; h < VOCABULARY_HASH_MAX; h++) {
		vocab_hash[h] = -1;
	}

	for(p = 0; p < pattern_max; p++) {
		h = hash_get(vocab[p]->word);

		while(vocab_hash[h] != -1) {
			h = (h + 1) % VOCABULARY_HASH_MAX;
		}

		vocab_hash[h] = p;
	}
}

static xWord** map_get(xWord** vocab, const dt_char* word) {
	dt_uint h = hash_get(word);
	
	while(1) {
		if(vocab_hash[h] == -1) {
			return NULL;
		}

		if(!strcmp(word, vocab[vocab_hash[h]]->word)) {
			return &vocab[vocab_hash[h]];
		}

		h = (h + 1) % VOCABULARY_HASH_MAX;
	}

	return NULL;
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

static xContext* context_insert(xContext* root, xWord* word, dt_int* success) {
	*success = 0;

	if(root) {
		dt_int cmp = strcmp(root->word->word, word->word);

		if(cmp > 0) {
			root->left = context_insert(root->left, word, success);
		} else if(cmp < 0) {
			root->right = context_insert(root->right, word, success);
		}

		return root;
	}

	*success = 1;
	return node_context_create(word);
}

static void context_flatten(xContext* root, xWord** arr, dt_int* index) {
	if(root) {
		context_flatten(root->left, arr, index);

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
		if(root->word->freq > 0) {
			arr[(*index)++] = root->word;
		}
#else
		arr[(*index)++] = root->word;
#endif

		context_flatten(root->right, arr, index);
	}
}

#if defined(FLAG_TEST_SIMILARITY) || defined(FLAG_TEST_CONTEXT)
#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
static dt_int filter_contains(xWord** filter, const dt_char* word) {
	dt_int p;

	for(p = 0; p < filter_max; p++) {
		if(!strcmp(filter[p]->word, word)) {
			return 1;
		}
	}

	return 0;
}
#endif
#endif

static dt_int list_contains(xWord* root, const dt_char* word) {
	while(root) {
		if(!strcmp(root->word, word)) {
			return 1;
		}

		root = root->next;
	}

	return 0;
}

#ifdef FLAT_FREE_MEMORY
static void list_release(xWord* root) {
	xWord* node;

	while(root) {
		node = root;
		root = root->next;
		node_release(node);
	}
}
#endif

static void context_release(xContext* root) {
	if(root) {
		context_release(root->left);
		context_release(root->right);
		node_context_release(root);
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
			root->freq++;
			node_release(*node);
			*node = root;
		}

		return root;
	}

	(*node)->freq++;
	*success = 1;
	return *node;
}

static void bst_flatten(xWord* root, xWord** arr, dt_int* index) {
	if(root) {
		bst_flatten(root->left, arr, index);

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
		if(root->freq > 0) {
			arr[root->index = (*index)++] = root;
		}
#else
		arr[root->index = (*index)++] = root;
#endif

		bst_flatten(root->right, arr, index);
	}
}

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
static void vocab_filter(xWord* corpus) {
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
static void vocab_save(xWord** vocab) {
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

static void vocab_map(xWord** vocab) {
	dt_int p;

	map_init(vocab);

	for(p = 0; p < pattern_max; p++) {
		*map_get(vocab, vocab[p]->word) = vocab[p];
	}
}

static void vocab_target(xWord** vocab) {
	dt_int p;

	for(p = 0; p < pattern_max; p++) {
		vocab[p]->target = (xWord**) calloc(vocab[p]->context_max, sizeof(xWord*));
		memcheck(vocab[p]->target);
		dt_int index = 0;
		context_flatten(vocab[p]->context, vocab[p]->target, &index);
		context_release(vocab[p]->context);
		vocab[p]->context = NULL;
		vocab[p]->context_max = index;
	}
}

#ifdef FLAG_NEGATIVE_SAMPLING
static void vocab_freq(xWord** vocab, dt_int* sum, dt_int* max) {
	dt_int p;

	for(*sum = *max = p = 0; p < pattern_max; p++) {
		*sum += vocab[p]->freq;
		*max = max(*max, vocab[p]->freq);
	}
}

#ifndef FLAG_MONTE_CARLO
static void vocab_sample(xWord** vocab) {
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

static xWord* node_create(const dt_char* word) {
	xWord* node = (xWord*) calloc(1, sizeof(xWord));
	memcheck(node);
	node->word = (dt_char*) calloc(strlen(word) + 2, sizeof(dt_char));
	memcheck(node->word);
	strcpy(node->word, word);
	node->word[strlen(node->word) + 1] = '*';
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
	if(root->word && root->word[strlen(root->word) + 1] == '*') {
		free(root->word);
		root->word = NULL;
	}

	if(root->target) {
		free(root->target);
		root->target = NULL;
	}

	root->left = root->right = root->next = NULL;
	root->index = root->prob = root->context_max = root->freq = 0;
	root->context = NULL;
	free(root);
}

static void node_context_release(xContext* root) {
	root->word = NULL;
	root->left = root->right = NULL;
	free(root);
}

static xWord* index_to_word(xWord** vocab, dt_int index) {
	return vocab[index];
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

static dt_int word_to_index(xWord** vocab, const dt_char* word) {
	xWord** ptr = map_get(vocab, word);

	if(ptr) {
		dt_int index = (*ptr)->index;

		if(index >= 0 && index < pattern_max) {
			return index;
		}
	}

#ifdef FLAG_LOG
	echo_fail("%s not found in corpus", word);
#endif

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

static void calculate_distribution() {
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

	for(p = 0; p <= FREQ_BUCKETS; p++) {
		buck = p == FREQ_BUCKETS ? pattern_max - 1 : p * span;

		echo_info("Sample #%d (%d/%d): %d occurrences", p + 1, buck, pattern_max, freqs[buck]);
	}

	free(freqs);

#ifdef FLAG_LOG
	echo_succ("Done calculating distribution");
#endif
}

static void resources_allocate() {
#ifdef FLAG_LOG
	echo("Allocating resources");
#endif

	dt_int i, k;

	vocab = (xWord**) calloc(pattern_max, sizeof(xWord*));
	memcheck(vocab);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "vocab", 1, pattern_max);
#endif

	vocab_hash = (dt_int*) calloc(VOCABULARY_HASH_MAX, sizeof(dt_int));
	memcheck(vocab_hash);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "vocab_hash", 1, VOCABULARY_HASH_MAX);
#endif

	patterns = (dt_int*) calloc(pattern_max, sizeof(dt_int));
	memcheck(patterns);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "patterns", 1, pattern_max);
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

	w_ho = (dt_float**) calloc(output_max, sizeof(dt_float*));
	memcheck(w_ho);
	for(k = 0; k < output_max; k++) {
		w_ho[k] = (dt_float*) calloc(hidden_max, sizeof(dt_float));
		memcheck(w_ho[k]);
	}
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "w_ho", hidden_max, output_max);
#endif

#ifdef FLAG_NEGATIVE_SAMPLING
#ifndef FLAG_MONTE_CARLO
#ifndef FLAG_UNIGRAM_DISTRIBUTION
	samples = (xWord**) calloc(pattern_max, sizeof(xWord*));
	memcheck(samples);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "samples", 1, pattern_max);
#endif
#endif
#endif
	output = (dt_float*) calloc(THREAD_MAX, sizeof(dt_float));
	memcheck(output);
#else
	dt_int t;

	output = (dt_float**) calloc(THREAD_MAX, sizeof(dt_float*));
	memcheck(output);
	for(t = 0; t < THREAD_MAX; t++) {
		output[t] = (dt_float*) calloc(output_max, sizeof(dt_float));
		memcheck(output[t]);
	}
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "output", 1, output_max);
#endif

	error = (dt_float**) calloc(THREAD_MAX, sizeof(dt_float*));
	memcheck(error);
	for(t = 0; t < THREAD_MAX; t++) {
		error[t] = (dt_float*) calloc(output_max, sizeof(dt_float));
		memcheck(error[t]);
	}
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "error", 1, output_max);
#endif
#endif

#ifdef FLAG_LOG
	echo_succ("Done allocating resources");
#endif
}

#ifdef FLAT_FREE_MEMORY
static void resources_release() {
	dt_int p, i, k;

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
	for(p = 0; p < filter_max; p++) {
		node_release(filter[p]);
	}
	free(filter);
	filter = NULL;
#endif

	for(p = 0; p < pattern_max; p++) {
		node_release(vocab[p]);
	}
	free(vocab);
	vocab = NULL;

	list_release(stops);
	stops = NULL;

	free(patterns);
	patterns = NULL;

	for(i = 0; i < input_max; i++) {
		free(w_ih[i]);
	}
	free(w_ih);
	w_ih = NULL;

	for(k = 0; k < output_max; k++) {
		free(w_ho[k]);
	}
	free(w_ho);
	w_ho = NULL;

#ifdef FLAG_NEGATIVE_SAMPLING
#ifndef FLAG_MONTE_CARLO
	free(samples);
	samples = NULL;
#endif
#else
	dt_int t;

	for(t = 0; t < THREAD_MAX; t++) {
		free(output[t]);
	}
	free(output);
	output = NULL;

	for(t = 0; t < THREAD_MAX; t++) {
		free(error[t]);
	}
	free(error);
	error = NULL;
#endif
}
#endif

static void initialize_corpus() {
#ifdef FLAG_LOG
	echo("Initializing corpus");
	clock_gettime(CLOCK_MONOTONIC, &time_start);
#endif

	FILE* fin = fopen(CORPUS_PATH, "r");
	FILE* fstop = fopen(STOP_PATH, "r");

	if(!fin || !fstop) {
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

	dt_int c, success, sent_end;
	const dt_char* sep = WORD_DELIMITERS;
	dt_char* tok;
	xWord* corpus;
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
						node->context = context_insert(node->context, window[c], &success);
						node->context_max += success;

						window[c]->context = context_insert(window[c]->context, node, &success);
						window[c]->context_max += success;

						token_max++;
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
	echo_info("Corpus size: %d words", token_max);
	echo_info("Vocabulary size: %d words", pattern_max);
	echo_info("Average word frequency: %lf", 1.0 * token_max / pattern_max);
#endif

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
#ifdef FLAG_LOG
	echo("Filtering vocabulary");
#endif

	vocab_filter(corpus);

#ifdef FLAG_LOG
	echo_succ("Done filtering vocabulary");
	echo_info("Vocabulary size: %d words", pattern_max);
#endif
#endif

	resources_allocate();

#ifdef FLAG_LOG
	echo("Flattening corpus");
#endif

	dt_int index = 0;
	bst_flatten(corpus, vocab, &index);

#ifdef FLAG_LOG
	echo_succ("Done flattening corpus");
#endif

#ifdef FLAG_BACKUP_VOCABULARY
#ifdef FLAG_LOG
	echo("Saving vocabulary");
#endif

	vocab_save(vocab);

#ifdef FLAG_LOG
	echo_succ("Done saving vocabulary");
#endif
#endif

	calculate_distribution();

#ifdef FLAG_LOG
	echo("Creating corpus map");
#endif

	vocab_map(vocab);

#ifdef FLAG_LOG
	echo_succ("Done creating corpus map");
#endif

#ifdef FLAG_LOG
	echo("Building word targets");
#endif

	vocab_target(vocab);

#ifdef FLAG_LOG
	echo_succ("Done building word targets");
#endif

#ifdef FLAG_NEGATIVE_SAMPLING
#ifdef FLAG_LOG
	echo("Calculating word frequency");
#endif

	vocab_freq(vocab, &corpus_freq_sum, &corpus_freq_max);

#ifdef FLAG_LOG
	echo_succ("Done calculating word frequency");
#endif

#ifndef FLAG_MONTE_CARLO
#ifdef FLAG_LOG
	echo("Creating sampling distribution");
#endif

	vocab_sample(vocab);

#ifdef FLAG_LOG
	echo_succ("Done creating sampling distribution");
#endif
#endif
#endif

	if(fclose(fin) == EOF || fclose(fstop) == EOF) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
	}

#ifdef FLAG_LOG
	echo_succ("Done initializing corpus (%d sec)", time_get(time_start));
#endif

#ifdef FLAG_INTERACTIVE_MODE
	echo("Entering interactive mode");

	dt_char cmd[LINE_CHARACTER_MAX] = { 0 };

	while(1) {
		echo("Command? (target, exit)");
		scanf("%s", cmd);

		if(!strcmp(cmd, "target")) {
			echo("Center word?");
			scanf("%s", cmd);

			dt_int index = word_to_index(vocab, cmd);

			if(!index_valid(index)) {
				continue;
			}

			xWord* center = index_to_word(vocab, index);

			for(c = 0; c < center->context_max; c++) {
				echo("Target #%d:\t%s", c + 1, center->target[c]->word);
			}
		} else if(!strcmp(cmd, "exit")) {
			break;
		} else {
			echo_fail(ERROR_COMMAND);
		}
	}

	echo_succ("Exiting interactive mode");
#endif
}

static void initialize_weights() {
#ifdef FLAG_LOG
	echo("Initializing weights");
#endif

	dt_int i, j, k;

	for(i = 0; i < input_max; i++) {
		for(j = 0; j < hidden_max; j++) {
#ifdef FLAG_FIXED_INITIAL_WEIGHTS
			w_ih[i][j] = INITIAL_WEIGHT_FIX;
#else
			w_ih[i][j] = random(INITIAL_WEIGHT_MIN, INITIAL_WEIGHT_MAX);
#endif
			w_ih[i][j] /= index_to_word(vocab, i)->freq;
		}
	}

	for(k = 0; k < output_max; k++) {
		for(j = 0; j < hidden_max; j++) {
			w_ho[k][j] = 0.0;
		}
	}

#ifdef FLAG_LOG
	echo_succ("Done initializing weights");
#endif
}

static void initialize_epoch(dt_int epoch) {
	dt_int p;

	for(p = 0; p < pattern_max; p++) {
		patterns[p] = p;
	}

	loss = 0;

#ifdef FLAG_FIXED_LEARNING_RATE
	alpha = LEARNING_RATE_FIX;
#else
	alpha = max(LEARNING_RATE_MIN, LEARNING_RATE_MAX * (1 - (dt_float) epoch / EPOCH_MAX));
#endif
}

#if defined(FLAG_TEST_CONTEXT) || !defined(FLAG_NEGATIVE_SAMPLING)
static void forward_propagate_input(dt_int index, dt_float* layer) {
	dt_int j, k;

	for(k = 0; k < output_max; k++) {
#ifdef FLAG_DROPOUT
		if(random(0, 1) < DROPOUT_RATE_MAX) {
			continue;
		}
#endif

		for(layer[k] = j = 0; j < hidden_max; j++) {
			layer[k] += w_ih[index][j] * w_ho[k][j];
		}
	}
}
#endif

#ifdef FLAG_NEGATIVE_SAMPLING
static dt_float sigmoid(dt_float x) {
	return 1.0 / (1.0 + exp(-x));
}

static void negative_sampling(xThread* t) {
	dt_int c, j, k, ck;
	dt_float e, delta_ih[hidden_max], delta_ho;
#ifdef FLAG_MONTE_CARLO
	dt_int exit;
	dt_float f, r, max_freq = ((dt_float) corpus_freq_max / corpus_freq_sum);
#endif
	for(c = 0; c < t->center->context_max; c++) {
		memset(delta_ih, 0, hidden_max * sizeof(dt_float));

		for(ck = 0; ck < NEGATIVE_SAMPLES_MAX; ck++) {
			if(ck) {
#ifdef FLAG_MONTE_CARLO
				for(exit = 0; exit < MONTE_CARLO_EMERGENCY; exit++) {
					k = random_int(0, pattern_max - 1);
					f = 1.0 * index_to_word(vocab, k)->freq / corpus_freq_sum;
					r = random(0, 1) * max_freq;

					if(k != t->center->target[c]->index && r > f) {
						break;
					}
				}
#else
#ifdef FLAG_UNIGRAM_DISTRIBUTION
				k = samples[random_int(0, corpus_freq_sum - 1)]->index;
#else
				k = samples[random_int(0, pattern_max - 1)]->index;
#endif
#endif
			} else {
				k = t->center->target[c]->index;
			}

			for(e = j = 0; j < hidden_max; j++) {
				e += w_ih[t->p][j] * w_ho[k][j];
			}

#ifdef FLAG_CALCULATE_LOSS
			loss -= log(sigmoid(ck ? -e : e));
#endif

			delta_ho = alpha * (sigmoid(e) - !ck);

			for(j = 0; j < hidden_max; j++) {
				delta_ih[j] += delta_ho * w_ho[k][j];
				w_ho[k][j] -= delta_ho * w_ih[t->p][j];
			}
		}

		for(j = 0; j < hidden_max; j++) {
			w_ih[t->p][j] -= delta_ih[j] / (1.0 * t->center->context_max);
		}
	}
}
#else
#ifdef FLAG_CALCULATE_LOSS
static void calculate_loss(xThread* t) {
	dt_int k, c;
	dt_float sum;

	for(sum = c = 0; c < t->center->context_max; c++) {
		sum += output[t->id][t->center->target[c]->index];
	}

	loss -= sum;

	for(sum = k = 0; k < output_max; k++) {
		sum += exp(output[t->id][k]);
	}

	loss += t->center->context_max * log(sum);
}
#endif
static void backward_propagate_error(xThread* t) {
	dt_int j, k, c;
	dt_float l;

	for(k = 0; k < output_max; k++) {
		for(error[t->id][k] = c = 0; c < t->center->context_max; c++) {
			error[t->id][k] += output[t->id][k] - (k == t->center->target[c]->index);
		}
	}

	for(j = 0; j < hidden_max; j++) {
		for(l = k = 0; k < output_max; k++) {
			l += error[t->id][k] * w_ho[k][j];
			w_ho[k][j] -= alpha * error[t->id][k] * w_ih[t->p][j];
		}

		w_ih[t->p][j] -= alpha * l;
	}
}
#endif

void nn_start() {
	static dt_int done = 0;

	if(done++) {
		return;
	}

	signal(SIGBUS, sigget);
	signal(SIGFPE, sigget);
	signal(SIGILL, sigget);
	signal(SIGSEGV, sigget);

	srand48(time(0));

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

#ifdef FLAT_FREE_MEMORY
	resources_release();
#endif

#ifdef FLAG_LOG_FILE
	if(fclose(flog) == EOF) {
		echo_fail(ERROR_FILE);
	}
#endif
}

void* thread_training_run(void* args) {
	xThread* t = (xThread*) args;
	dt_int epoch, p1, p2, progress;
	dt_int from = t->id * pattern_max / THREAD_MAX;
	dt_int to = t->id == THREAD_MAX - 1 ? pattern_max : from + pattern_max / THREAD_MAX;

	for(epoch = 0; epoch < EPOCH_MAX; epoch++) {
		if(!t->id) {
#ifdef FLAG_LOG
			echo("Started epoch %d/%d", epoch + 1, EPOCH_MAX);
			clock_gettime(CLOCK_MONOTONIC, &time_start);
#endif

			initialize_epoch(epoch);
		}

		if(epoch) {
			pthread_mutex_lock(&mtx_count_epoch);

			if(!--count_epoch) {
				sem_wait(sem_epoch_1);
				sem_post(sem_epoch_2);
			}

			pthread_mutex_unlock(&mtx_count_epoch);

			sem_wait(sem_epoch_2);
			sem_post(sem_epoch_2);
		}

		for(p1 = from; p1 < to; p1++) {
#ifdef FLAG_LOG
			if(!t->id) {
				if(!((to - p1) % LOG_PERIOD_PASS)) {
					for(progress = p2 = 0; p2 < pattern_max; p2++) {
						progress = patterns[p2] < 0 ? progress + 1 : progress;
					}

					echo("Training %d/%d", progress, pattern_max);
				}
			}
#endif

			t->center = index_to_word(vocab, t->p = patterns[p1]);
			patterns[p1] = -1;

			vector_normalize(w_ih[t->p], hidden_max);

#ifdef FLAG_NEGATIVE_SAMPLING
			negative_sampling(t);
#else
			forward_propagate_input(t->p, output[t->id]);
			vector_softmax(output[t->id], output_max);
#ifdef FLAG_CALCULATE_LOSS
			calculate_loss(t);
#endif
			backward_propagate_error(t);
#endif
		}

		pthread_mutex_lock(&mtx_count_epoch);

		if(++count_epoch == THREAD_MAX) {
			sem_wait(sem_epoch_2);
			sem_post(sem_epoch_1);
		}

		pthread_mutex_unlock(&mtx_count_epoch);

		sem_wait(sem_epoch_1);
		sem_post(sem_epoch_1);

		if(!t->id) {
#ifdef FLAG_BACKUP_WEIGHTS
			weights_save();
#endif

#ifdef FLAG_LOG
			echo_succ("Finished epoch %d/%d (%d sec)", epoch + 1, EPOCH_MAX, time_get(time_start));
#ifdef FLAG_CALCULATE_LOSS
			echo_info("Loss %lf", loss);
#endif
#endif
		}
	}

	return NULL;
}

void training_run() {
#ifdef FLAG_LOG
	struct timespec time_local;
	clock_gettime(CLOCK_MONOTONIC, &time_local);
	echo("Started training");
#endif

	pthread_mutex_init(&mtx_count_epoch, NULL);

	sem_epoch_1 = sem_open("/sem_epoch_1", O_CREAT, 0666, 0);
	sem_epoch_2 = sem_open("/sem_epoch_2", O_CREAT, 0666, 1);

	dt_int t;

	for(t = 0; t < THREAD_MAX; t++) {
		(thread_args + t)->id = t;
		pthread_create(thread_ids + t, NULL, thread_training_run, thread_args + t);
	}

	for(t = 0; t < THREAD_MAX; t++) {
		pthread_join(thread_ids[t], NULL);
	}

	pthread_mutex_destroy(&mtx_count_epoch);

	sem_unlink("/sem_epoch_1");
	sem_close(sem_epoch_1);

	sem_unlink("/sem_epoch_2");
	sem_close(sem_epoch_2);

#ifdef FLAG_LOG
	echo_succ("Finished training (%d sec)", time_get(time_local));
#endif
}

#ifdef FLAG_TEST_SIMILARITY
void test_similarity() {
#ifdef FLAG_LOG
	struct timespec time_local;
	clock_gettime(CLOCK_MONOTONIC, &time_local);
	echo("Started similarity testing");
#endif

	FILE* ftest = fopen(TEST_PATH, "r");

	if(!ftest) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
		return;
	}

	dt_char line[LINE_CHARACTER_MAX];
	dt_int sent_end;

	while(fgets(line, LINE_CHARACTER_MAX, ftest)) {
		line[strlen(line) - 1] = '\0';
		word_clean(line, &sent_end);

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
		dt_int skip = word_stop(line) || filter_contains(filter, line);
#else
		dt_int skip = word_stop(line);
#endif

		if(skip) {
#ifdef FLAG_LOG
			echo_info("Skipping word %s", line);
#endif
			continue;
		}

#ifdef FLAG_LOG
		echo_info("Word: %s", line);
#endif

		dt_int p, index = word_to_index(vocab, line);

		if(!index_valid(index)) {
			return;
		}

		xWord* dist[pattern_max];

		for(p = 0; p < pattern_max; p++) {
			dist[p] = vocab[p];
			vector_distance(w_ih[index], w_ih[p], hidden_max, &dist[p]->dist);
		}

		qsort(dist, pattern_max, sizeof(xWord*), cmp_dist);

		for(index = p = 1; p <= PREDICTION_MAX; p++) {
#ifdef FLAG_LOG
			echo("#%d\t%lf\t%s", index++, dist[p]->dist, dist[p]->word);
#endif
		}
	}

	if(fclose(ftest) == EOF) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
	}

#ifdef FLAG_LOG
	echo_succ("Finished similarity testing (%d sec)", time_get(time_local));
#endif
}
#endif

#ifdef FLAG_TEST_CONTEXT
void test_context() {
#ifdef FLAG_LOG
	struct timespec time_local;
	clock_gettime(CLOCK_MONOTONIC, &time_local);
	echo("Started context testing");
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

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
		dt_int skip = word_stop(line) || filter_contains(filter, line);
#else
		dt_int skip = word_stop(line);
#endif

		if(skip) {
#ifdef FLAG_LOG
			echo_info("Skipping word %s", line);
#endif
			continue;
		}

		dt_int count = PREDICTION_MAX;

#ifdef FLAG_LOG
		echo_info("Center: %s", line);
		clock_gettime(CLOCK_MONOTONIC, &time_start);
#endif

		dt_int c, k, s, index = word_to_index(vocab, line);

		if(!index_valid(index)) {
			return;
		}

#ifdef FLAG_NEGATIVE_SAMPLING
		forward_propagate_input(index, output);
		vector_softmax(output, output_max);
#else
		forward_propagate_input(index, output[0]);
		vector_softmax(output[0], output_max);
#endif

		xWord* pred[pattern_max];

		for(k = 0; k < output_max; k++) {
			pred[k] = vocab[k];
#ifdef FLAG_NEGATIVE_SAMPLING
			pred[k]->prob = output[k];
#else
			pred[k]->prob = output[0][k];
#endif
		}

		qsort(pred, pattern_max, sizeof(xWord*), cmp_prob);

		xWord* center = index_to_word(vocab, index);

		for(success = 0, index = 1, k = 0; k < count; k++) {
			if(!strcmp(pred[k]->word, line)) {
				count++;
				continue;
			}

			for(s = c = 0; c < center->context_max; c++) {
				if(!strcmp(center->target[c]->word, pred[k]->word)) {
					if(index == 1) {
						success = 1;
					}

					s = 1;
					break;
				}
			}

#ifdef FLAG_LOG
			echo("#%d\t%s\t%lf\t%s", index++, s ? "OK" : ".", pred[k]->prob, pred[k]->word);
#endif
		}

#ifdef FLAG_LOG
		echo_cond(success, "Prediction %scorrect (%d sec)", success ? "" : "not ", time_get(time_start));
#endif
		test_count++, tries_sum += success;
	}

	if(fclose(ftest) == EOF) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
	}

#ifdef FLAG_LOG
	dt_float prec = 100.0 * tries_sum / test_count;
	echo_cond(prec > 50.0, "Precision: %.1lf%%", prec);
	echo_succ("Finished context testing (%d sec)", time_get(time_local));
#endif
}
#endif

void testing_run() {
#ifdef FLAG_TEST_SIMILARITY
	test_similarity();
#endif

#ifdef FLAG_TEST_CONTEXT
	test_context();
#endif
}

void weights_save() {
#ifdef FLAG_LOG
	echo("Started saving weights");
#endif

#ifdef FLAG_BINARY_OUTPUT
	FILE* fwih = fopen(WEIGHTS_IH_PATH, "wb");
	FILE* fwho = fopen(WEIGHTS_HO_PATH, "wb");
#else
	FILE* fwih = fopen(WEIGHTS_IH_PATH, "w");
	FILE* fwho = fopen(WEIGHTS_HO_PATH, "w");
#endif

	if(!fwih || !fwho) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
		return;
	}

	dt_int i, j, k;

	for(i = 0; i < input_max; i++) {
#ifdef FLAG_BINARY_OUTPUT
		fwrite(w_ih[i], sizeof(dt_float), hidden_max, fwih);
#else
		for(j = 0; j < hidden_max; j++) {
			fprintf(fwih, j ? "\t%lf" : "%lf", w_ih[i][j]);
		}

		fprintf(fwih, "\n");
#endif
	}

	for(k = 0; k < output_max; k++) {
#ifdef FLAG_BINARY_OUTPUT
		fwrite(w_ho[k], sizeof(dt_float), output_max, fwho);
#else
		for(j = 0; j < hidden_max; j++) {
			fprintf(fwho, j ? "\t%lf" : "%lf", w_ho[k][j]);
		}

		fprintf(fwho, "\n");
#endif
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

#ifdef FLAG_BINARY_INPUT
	FILE* fwih = fopen(WEIGHTS_IH_PATH, "rb");
	FILE* fwho = fopen(WEIGHTS_HO_PATH, "rb");
#else
	FILE* fwih = fopen(WEIGHTS_IH_PATH, "r");
	FILE* fwho = fopen(WEIGHTS_HO_PATH, "r");
#endif

	if(!fwih || !fwho) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
		return;
	}

	dt_int i, j, k;

	for(i = 0; i < input_max; i++) {
#ifdef FLAG_BINARY_INPUT
		fread(w_ih[i], sizeof(dt_float), hidden_max, fwih);
#else
		for(j = 0; j < hidden_max; j++) {
			fscanf(fwih, "%lf", &w_ih[i][j]);
		}
#endif
	}

	for(k = 0; k < output_max; k++) {
#ifdef FLAG_BINARY_INPUT
		fread(w_ho[k], sizeof(dt_float), output_max, fwho);
#else
		for(j = 0; j < hidden_max; j++) {
			fscanf(fwho, "%lf", &w_ho[k][j]);
		}
#endif
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

	dt_int index, sent_end, j;
	const dt_char* sep = WORD_DELIMITERS;
	dt_char* tok = strtok(sentence, sep);

	while(tok) {
		word_clean(tok, &sent_end);

		if(!word_stop(tok)) {
			index = word_to_index(vocab, tok);

			if(!index_valid(index)) {
				continue;
			}

			for(j = 0; j < hidden_max; j++) {
				vector[j] += w_ih[index][j];
			}
		}

		tok = strtok(NULL, sep);
	}

	vector_normalize(vector, hidden_max);
}
