#include "include/nn.h"

static clock_t elapsed;

static int input_max, hidden_max, output_max, pattern_max;
static int i, j, k, c;
static int p, p1, p2;

static double alpha;

static double sum, loss;
static int epoch, initialized;

static xBit* input;
static double* hidden;
static double* output;
static double* output_raw;
static int** target;
static double** weight_ih;
static double** weight_ho;
static int* training;
static double* error;

static xWord* root;
static char context[SENTENCE_MAX][WORD_MAX][CHARACTER_MAX];
static xBit onehot[SENTENCE_MAX * WORD_MAX][SENTENCE_MAX * WORD_MAX];
static char* test_word;

static FILE* flog;
static FILE* ffilter;

static xBit* map_get(const char* word) {
	unsigned int hash = 0, c;

	for(size_t i = 0; word[i]; i++) {
		c = (unsigned char) word[i];
		hash = (hash << 3) + (hash >> (sizeof(hash) * CHAR_BIT - 3)) + c;
	}

	return onehot[hash % (SENTENCE_MAX * WORD_MAX)];
}

static xWord* bst_insert(xWord* node, const char* word, int* success) {
	if(!node) {
		node = (xWord*) calloc(1, sizeof(xWord));
		strcpy(node->word, word);
		node->left = node->right = NULL;
		node->prob = 0.0;
		node->context_count = 0;
		*success = 1;
		return node;
	}

	int cmp = strcmp(word, node->word);

	if(cmp < 0) {
		node->left = bst_insert(node->left, word, success);
	} else if(cmp > 0) {
		node->right = bst_insert(node->right, word, success);
	}

	return node;
}

static void bst_to_map(xWord* node, int* index) {
	if(node) {
		bst_to_map(node->left, index);
		map_get(node->word)[(*index)++].on = 1;
		bst_to_map(node->right, index);
	}
}

static void bst_free(xWord* node) {
	if(node) {
		bst_free(node->left);
		bst_free(node->right);
		node->left = NULL;
		node->right = NULL;
		free(node);
		node->context_count = 0;
		node = NULL;
	}
}

static xWord* bst_get(xWord* node, int* index) {
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

static int filter_word(char* word) {
	if(strlen(word) < 2) {
		return 1;
	}

	if(!ffilter) {
		fprintf(flog, FILE_ERROR_MESSAGE);
		return 0;
	}

	char line[CHARACTER_MAX];

	fseek(ffilter, 0, SEEK_SET);

	while(fgets(line, CHARACTER_MAX, ffilter)) {
		line[strlen(line) - 1] = '\0';

		if(!strcmp(line, word)) {
			return 1;
		}
	}

	return 0;
}

static void parse_corpus_file() {
	FILE* fin = fopen(CORPUS_PATH, "r");

	if(!fin) {
		fprintf(flog, FILE_ERROR_MESSAGE);
		return;
	}

	int i = 0, j = 0, success;
	char c, word[WORD_MAX] = { 0 };
	char* pw = word;

	while((c = fgetc(fin)) != EOF) {
		if(isalnum(c) || c == '-') {
			*pw++ = tolower(c);
		} else if(!(isalnum(c) || c == '-') && word[0]) {
			if(!filter_word(word)) {
				strcpy(context[i][j++], word);
				success = 0;
				root = bst_insert(root, word, &success);

				if(success) {
					pattern_max = input_max = ++output_max;
					hidden_max = WINDOW_MAX * 2;
				}
			}else {
				printf("%s\n", word);
			}

			memset(pw = word, 0, sizeof(word));
		} else if(c == '.') {
			++i, j = 0;
		}
	}

	if(fclose(fin) == EOF) {
		fprintf(flog, FILE_ERROR_MESSAGE);
	}
}

static void allocate_layers() {
	input = (xBit*) calloc(input_max, sizeof(xBit));
	hidden = (double*) calloc(hidden_max, sizeof(double));
	output = (double*) calloc(output_max, sizeof(double));
	output_raw = (double*) calloc(output_max, sizeof(double));

	target = (int**) calloc(pattern_max, sizeof(int*));
	for(p = 0; p < pattern_max; p++) {
		target[p] = (int*) calloc(output_max, sizeof(int));
		memset(target[p], -1, output_max);
	}

	weight_ih = (double**) calloc(input_max, sizeof(double*));
	for(i = 0; i < input_max; i++) {
		weight_ih[i] = (double*) calloc(hidden_max, sizeof(double));
	}

	weight_ho = (double**) calloc(hidden_max, sizeof(double*));
	for(j = 0; j < hidden_max; j++) {
		weight_ho[j] = (double*) calloc(output_max, sizeof(double));
	}

	training = (int*) calloc(pattern_max, sizeof(int));
	error = (double*) calloc(output_max, sizeof(double));
}

static void free_layers() {
	free(input);
	free(hidden);
	free(output);
	free(output_raw);

	for(p = 0; p < pattern_max; p++) {
		free(target[p]);
	}
	free(target);

	for(i = 0; i < input_max; i++) {
		free(weight_ih[i]);
	}
	free(weight_ih);

	for(j = 0; j < hidden_max; j++) {
		free(weight_ho[j]);
	}
	free(weight_ho);

	free(training);
	free(error);
}

static void initialize_training() {
	ffilter = fopen(FILTER_PATH, "r");

	parse_corpus_file();
	allocate_layers();

	int index = 0;

	bst_to_map(root, &index);

	for(i = 0; i < SENTENCE_MAX; i++) {
		for(j = 0; j < WORD_MAX; j++) {
			if(!context[i][j][0]) {
				continue;
			}

			xBit* word = map_get(context[i][j]);

			for(index = 0; index < pattern_max && !word[index].on; index++)
				;

			int index_copy = index;
			xWord* center_node = bst_get(root, &index_copy);

			for(k = j - WINDOW_MAX; k <= j + WINDOW_MAX; k++) {
				if(k == j || k < 0 || !context[i][k][0]) {
					continue;
				}

				word = map_get(context[i][k]);

				int pom;

				for(pom = 0; pom < output_max; pom++) {
					if(word[pom].on) {
						int found = 0;

						for(p = 0; p < center_node->context_count; p++) {
							if(target[index][p] == pom) {
								found = 1;
								break;
							}
						}

						if(!found) {
							target[index][(center_node->context_count)++] = pom;
						}
					}
				}
			}
		}
	}

	for(p = 0; p < pattern_max; p++) {
		training[p] = p;
	}

	flog = fopen(LOG_PATH, "w");
}

static void initialize_test() {
	printf("\nCenter:\t\t%s\n", test_word);

	xBit* onehot = map_get(test_word);

	int index, curr = 0;
	p = 0;

	for(i = 0; i < input_max; i++) {
		input[i].on = onehot[i].on;

		curr += onehot[i].on * i;
	}

	int curr_copy = curr;
	xWord* center_word = bst_get(root, &curr_copy);

	for(k = 0; k < center_word->context_count; k++) {
		index = target[curr][k];
		xWord* context = bst_get(root, &index);
		printf("Context #%d:\t%s\n", k, context->word);
	}

	printf("\n");
}

static void initialize_weights() {
	if(initialized) {
		return;
	}

	for(i = 0; i < input_max; i++) {
		for(j = 0; j < hidden_max; j++) {
			weight_ih[i][j] = 2.0 * (random() - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}

	for(j = 0; j < hidden_max; j++) {
		for(k = 0; k < output_max; k++) {
			weight_ho[j][k] = 2.0 * (random() - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}
}

static void initialize_epoch() {
	for(p = 0; p < pattern_max; p++) {
		p1 = p + random() * (pattern_max - p);

		p2 = training[p];
		training[p] = training[p1];
		training[p1] = p2;
	}

	alpha = max(LEARNING_RATE_MIN, LEARNING_RATE * (1 - (double) epoch / EPOCH_MAX));

	loss = 0.0;
}

static void initialize_input() {
	p = training[p1];

	for(i = 0; i < input_max; i++) {
		input[i].on = 0;
	}

	input[p].on = 1;
}

static void forward_propagate_input_layer() {
	for(j = 0; j < hidden_max; j++) {
		hidden[j] = 0.0;

		for(i = 0; i < input_max; i++) {
			hidden[j] += input[i].on * weight_ih[i][j];
		}
	}
}

static void forward_propagate_hidden_layer() {
	for(k = 0; k < output_max; k++) {
		output[k] = 0.0;

		for(j = 0; j < hidden_max; j++) {
			output[k] += hidden[j] * weight_ho[j][k];
		}
	}
}

static void normalize_output_layer() {
	double out_max = DBL_MIN;

	for(k = 0; k < output_max; k++) {
		out_max = max(out_max, output_raw[k] = output[k]);
	}

	double out_exp[output_max];
	sum = 0.0;

	for(k = 0; k < output_max; k++) {
		sum += out_exp[k] = exp(output[k] - out_max);
	}

	for(k = 0; k < output_max; k++) {
		output[k] = out_exp[k] / sum;
	}
}

static void calculate_error() {
	int p_copy = p;
	xWord* center_node = bst_get(root, &p_copy);

	int context_max = center_node->context_count;
	double error_t[context_max][output_max];

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
}

static void update_hidden_layer_weights() {
	for(j = 0; j < hidden_max; j++) {
		for(k = 0; k < output_max; k++) {
			weight_ho[j][k] -= alpha * hidden[j] * error[k];
		}
	}
}

static void update_input_layer_weights() {
	double error_t[output_max];

	for(j = 0; j < hidden_max; j++) {
		error_t[j] = 0.0;

		for(k = 0; k < output_max; k++) {
			error_t[j] += error[k] * weight_ho[j][k];
		}
	}

	for(i = 0; i < input_max; i++) {
		for(j = 0; j < hidden_max; j++) {
			weight_ih[i][j] -= alpha * input[k].on * error_t[j];
		}
	}
}

static void calculate_loss() {
	int p_copy = p;
	xWord* center_node = bst_get(root, &p_copy);

	int context_max = center_node->context_count;

	for(c = 0; c < context_max; c++) {
		loss -= output_raw[target[p][c]];
	}

	sum = 0.0;

	for(k = 0; k < output_max; k++) {
		sum += exp(output_raw[k]);
	}

	loss += context_max * log(sum);
}

static void log_epoch() {
	if(epoch % LOG_PERIOD) {
		return;
	}

	fprintf(flog, "%cEpoch\t%d\n", epoch ? '\n' : '\0', epoch + 1);
	fprintf(flog, "Took\t%lf sec\n", (double) (elapsed = clock() - elapsed) / CLOCKS_PER_SEC);
	fprintf(flog, "Loss\t%lf\n", loss);
}

static int cmp_words(const void* a, const void* b) {
	double diff = (*(xWord*) a).prob - (*(xWord*) b).prob;

	return diff < 0 ? 1 : diff > 0 ? -1 : 0;
}

void start_training() {
	srand(time(0));

	initialize_training();
	initialize_weights();

	elapsed = clock();

	for(epoch = 0; epoch < EPOCH_MAX; epoch++) {
		printf("Epoch:\t%d / %d\n", epoch + 1, EPOCH_MAX);

		initialize_epoch();

		for(p1 = 0; p1 < pattern_max; p1++) {
			initialize_input();
			forward_propagate_input_layer();
			forward_propagate_hidden_layer();
			normalize_output_layer();
			calculate_error();
			update_hidden_layer_weights();
			update_input_layer_weights();
			calculate_loss();
		}

		if(LOG_EPOCH) {
			log_epoch();
		}
	}
}

void finish_training() {
	bst_free(root);
	free_layers();
	fclose(flog);
	fclose(ffilter);
}

void get_predictions(char* word, int count) {
	test_word = word;

	initialize_test();
	forward_propagate_input_layer();
	forward_propagate_hidden_layer();
	normalize_output_layer();

	xWord pred[output_max];

	for(k = 0; k < output_max; k++) {
		int index = k;
		pred[k] = *bst_get(root, &index);
		pred[k].prob = output[k];
	}

	qsort(pred, output_max, sizeof(xWord), cmp_words);

	int index;

	for(index = 1, k = 0; k < count; k++, index++) {
		if(!strcmp(pred[k].word, word)) {
			count++;
			continue;
		}

		printf("#%d\t%lf\t%s\n", index, pred[k].prob, pred[k].word);
	}
}

void save_weights() {
	FILE* fwih = fopen(WEIGHTS_IH_PATH, "w");
	FILE* fwho = fopen(WEIGHTS_HO_PATH, "w");

	if(!fwih || !fwho) {
		fprintf(flog, FILE_ERROR_MESSAGE);
		return;
	}

	for(i = 0; i < input_max; i++) {
		for(j = 0; j < hidden_max; j++) {
			fprintf(fwih, "%s%lf", j ? " " : "", weight_ih[i][j]);
		}

		fprintf(fwih, "\n");
	}

	for(j = 0; j < hidden_max; j++) {
		for(k = 0; k < output_max; k++) {
			fprintf(fwho, "%s%lf", k ? " " : "", weight_ho[j][k]);
		}

		fprintf(fwho, "\n");
	}

	fclose(fwih);
	fclose(fwho);
}

void load_weights() {
	FILE* fwih = fopen(WEIGHTS_IH_PATH, "w");
	FILE* fwho = fopen(WEIGHTS_HO_PATH, "w");

	if(!fwih || !fwho) {
		fprintf(flog, FILE_ERROR_MESSAGE);
		return;
	}

	for(i = 0; i < input_max; i++) {
		for(j = 0; j < hidden_max; j++) {
			fscanf(fwih, "%lf", &weight_ih[i][j]);
		}
	}

	for(j = 0; j < hidden_max; j++) {
		for(k = 0; k < output_max; k++) {
			fscanf(fwho, "%lf", &weight_ho[j][k]);
		}
	}

	fclose(fwih);
	fclose(fwho);

	initialized = 1;
}
