#include "nn.h"

static dt_ull token_max;

static dt_float alpha;
static dt_float loss;

static dt_int* patterns;
static dt_float** w_ih;
static dt_float** w_ho;
static dt_float* output;

static xThread thread_args[THREAD_MAX];
static pthread_t thread_ids[THREAD_MAX];
static pthread_mutex_t mtx_count_epoch;
static sem_t* sem_epoch_1;
static sem_t* sem_epoch_2;
static dt_int count_epoch;

// Allocate auxiliary structures
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

	vocab_hash = (dt_ull*) calloc(VOCABULARY_HASH_MAX, sizeof(dt_ull));
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

	samples = (xWord**) calloc(pattern_max, sizeof(xWord*));
	memcheck(samples);

#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "samples", 1, pattern_max);
#endif

	output = (dt_float*) calloc(THREAD_MAX, sizeof(dt_float));
	memcheck(output);
#ifdef FLAG_LOG
	echo_info("Dimension of %s: %dx%d", "output", 1, output_max);
#endif

#ifdef FLAG_LOG
	echo_succ("Done allocating resources");
#endif
}

#ifdef FLAG_FREE_MEMORY
// Release allocaed auxiliary structures
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

	free(samples);
	samples = NULL;
}
#endif

// Initialize vocabulary BST by reading corpus file
static void initialize_corpus() {
#ifdef FLAG_LOG
	echo("Initializing corpus");
	clock_gettime(CLOCK_MONOTONIC, &time_start);
#endif

	FILE* fin = fopen(TRAIN_PATH, "r");
	FILE* fstop = fopen(STOP_PATH, "r");

	if(!fin || !fstop) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
		exit(1);
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
	xWord* corpus = NULL;
	xWord* window[WINDOW_MAX] = { 0 };

#ifdef FLAG_LOG
	echo("Reading corpus file");
#endif

	while(fgets(line, LINE_CHARACTER_MAX, fin)) {
		tok = strtok(line, sep);

		while(tok) {
			word_clean(tok, &sent_end);

#ifdef FLAG_FILTER_VOCABULARY_STOP
			if(!word_stop(tok)) {
#endif
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
#ifdef FLAG_FILTER_VOCABULARY_STOP
			}
#endif

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

#ifdef FLAG_LOG
	echo("Saving vocabulary");
#endif

	vocab_save(vocab);

#ifdef FLAG_LOG
	echo_succ("Done saving vocabulary");
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

#ifdef FLAG_LOG
	echo("Calculating word frequency");
#endif

	vocab_freq(vocab, &corpus_freq_sum, &corpus_freq_max);

#ifdef FLAG_LOG
	echo_succ("Done calculating word frequency");
#endif

#ifdef FLAG_LOG
	echo("Creating sampling distribution");
#endif

	vocab_sample(vocab);

#ifdef FLAG_LOG
	echo_succ("Done creating sampling distribution");
#endif

	if(fclose(fin) == EOF || fclose(fstop) == EOF) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
	}

#ifdef FLAG_LOG
	echo_succ("Done initializing corpus (%d sec)", time_get(time_start));
#endif
}

// Initialize weights between layers
static void initialize_weights() {
#ifdef FLAG_LOG
	echo("Initializing weights");
#endif

	dt_int p, j;

	for(p = 0; p < pattern_max; p++) {
		for(j = 0; j < hidden_max; j++) {
			w_ih[p][j] = xavier(pattern_max, pattern_max);
			w_ho[p][j] = xavier(pattern_max, pattern_max);
		}
	}

#ifdef FLAG_LOG
	echo_succ("Done initializing weights");
#endif
}

// Initialize parameters for new training epoch
static void initialize_epoch(dt_int epoch) {
	dt_int p;

	for(loss = p = 0; p < pattern_max; p++) {
		patterns[p] = p;
	}

	alpha = max(LEARNING_RATE_MIN, LEARNING_RATE_MAX * (1 - (dt_float) epoch / EPOCH_MAX));
}

#ifdef FLAG_TEST_CONTEXT
// Forward propagate word through network
static void forward_propagate_input(dt_int index, dt_float* layer) {
	dt_int j, k;

	for(k = 0; k < output_max; k++) {
		for(layer[k] = j = 0; j < hidden_max; j++) {
			layer[k] += w_ih[index][j] * w_ho[k][j];
		}
	}
}
#endif

// Perform negative sampling
static void negative_sampling(xThread* t) {
	dt_int c, j, k, ck, tf;
	dt_float e, delta_ih[hidden_max], delta_ho;

	for(c = 0; c < t->center->context_max; c++) {
		for(tf = 0; tf < t->center->target_freq[c]; tf++) {
			memset(delta_ih, 0, hidden_max * sizeof(dt_float));
			
			for(ck = 0; ck < SAMPLE_MAX; ck++) {
				if(ck) {
					k = samples[random_int(0, pattern_max - 1)]->index;
				} else {
					k = t->center->target[c]->index;
				}

				for(e = j = 0; j < hidden_max; j++) {
					e += w_ih[t->p][j] * w_ho[k][j];
				}

				loss -= log(sigmoid(ck ? -e : e));
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
}

// Initialize everything
void nn_start() {
	static dt_int done = 0;

	if(done++) {
		return;
	}

	signal(SIGBUS, sigget);
	signal(SIGFPE, sigget);
	signal(SIGILL, sigget);
	signal(SIGSEGV, sigget);

	srand(0);

#ifdef FLAG_LOG_FILE
	flog = fopen(LOG_PATH, "w");
#endif

	initialize_corpus();
	initialize_weights();
}

// Clear everything
void nn_finish() {
	static dt_int done = 0;

	if(done++) {
		return;
	}

	invalid_index_print();

#ifdef FLAG_FREE_MEMORY
	resources_release();
#endif

#ifdef FLAG_LOG_FILE
	if(fclose(flog) == EOF) {
#ifdef FLAG_LOG
		echo_fail(ERROR_FILE);
#endif
	}
#endif
}

// Thread function for training
void* thread_training_run(void* args) {
	xThread* t = (xThread*) args;
	dt_int epoch, p1;
	dt_int from = t->id * pattern_max / THREAD_MAX;
	dt_int to = t->id == THREAD_MAX - 1 ? pattern_max : from + pattern_max / THREAD_MAX;

#ifdef FLAG_LOG
	dt_int p2, progress;
#endif

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
			negative_sampling(t);
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
			weights_save();

#ifdef FLAG_LOG
			echo_succ("Finished epoch %d/%d (%d sec)", epoch + 1, EPOCH_MAX, time_get(time_start));
			echo_info("Loss %lf", loss);
#endif
		}
	}

	return NULL;
}

// Run multithreaded training
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
// Perform similarity test for words in test file
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

		dt_int skip = 0;

#ifdef FLAG_FILTER_VOCABULARY_STOP
		skip = skip || word_stop(line);
#endif

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
		skip = skip || filter_contains(filter, line);
#endif

		if(skip) {
#ifdef FLAG_LOG
			echo_info("Skipping word: %s", line);
#endif
			continue;
		}

#ifdef FLAG_LOG
		echo_info("Word: %s", line);
#endif

		dt_int p, index = word_to_index(vocab, line);

		if(!index_valid(index)) {
			continue;
		}

		xWord* dist[pattern_max];

		for(p = 0; p < pattern_max; p++) {
			dist[p] = vocab[p];
			vector_distance(w_ih[index], w_ih[p], hidden_max, &dist[p]->dist);
		}

		qsort(dist, pattern_max, sizeof(xWord*), cmp_dist);

		for(p = 1; p <= PREDICTION_MAX; p++) {
#ifdef FLAG_LOG
			echo("#%d\t%lf\t%s", p, dist[p]->dist, dist[p]->word);
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
// Perform context test for words in test file
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

		dt_int skip = 0;

#ifdef FLAG_FILTER_VOCABULARY_STOP
		skip = skip || word_stop(line);
#endif

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
		skip = skip || filter_contains(filter, line);
#endif

		if(skip) {
#ifdef FLAG_LOG
			echo_info("Skipping word: %s", line);
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
			continue;
		}

		forward_propagate_input(index, output);
		vector_softmax(output, output_max);

		xWord* pred[pattern_max];

		for(k = 0; k < output_max; k++) {
			pred[k] = vocab[k];
			pred[k]->prob = output[k];
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

#ifdef FLAG_TEST_ORTHANT
// Perform orthant test for words in test file
void test_orthant() {
#ifdef FLAG_LOG
	struct timespec time_local;
	clock_gettime(CLOCK_MONOTONIC, &time_local);
	echo("Started orthant testing");
#endif

	dt_int j, i, count;

	for(j = 0; j < hidden_max; j++) {
		for(count = i = 0; i < input_max; i++) {
			count += w_ih[i][j] < 0.0;
		}

		echo_info("Coordinate #%d: %d/%d", j + 1, count, pattern_max - count);
	}

#ifdef FLAG_LOG
	echo_succ("Finished orthant testing (%d sec)", time_get(time_local));
#endif
}
#endif

// Run testing
void testing_run() {
#ifdef FLAG_TEST_SIMILARITY
	test_similarity();
#endif

#ifdef FLAG_TEST_CONTEXT
	test_context();
#endif

#ifdef FLAG_TEST_ORTHANT
	test_orthant();
#endif
}

// Save weights to file
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

	dt_int i, k;
#ifndef FLAG_BINARY_OUTPUT
	dt_int j;
#endif

	for(i = 0; i < input_max; i++) {
		vector_normalize(w_ih[i], hidden_max);

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

// Load weights from file
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

	dt_int i, k;
#ifndef FLAG_BINARY_INPUT
	dt_int j;
#endif

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
