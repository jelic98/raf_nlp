#include "include/nn.h"

static clock_t elapsed;

static int pattern_max, input_max, hidden_max, output_max;
static int i, j, k;
static int p, p1, p2;

static double max, sum, offset, error;
static int epoch, count;

static xBit** input;
static xBit** target;
static double** hidden;
static double** output;
static double** weight_ih;
static double** weight_ho;
static int* training;
static double* error_d;

static xWord* root = NULL;
static char context[SENTENCE_MAX][WORD_MAX][CHARACTER_MAX];
static xBit onehot[SENTENCE_MAX * WORD_MAX][SENTENCE_MAX * WORD_MAX];

static FILE* fout;
static FILE* flog;

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
		node->count = 1;
		*success = 1;
		return node;
	}

	int cmp = strcmp(word, node->word);

	if(cmp < 0) {
		node->left = bst_insert(node->left, word, success);
	} else if(cmp > 0) {
		node->right = bst_insert(node->right, word, success);
	} else {
		node->count++;
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
		node->count = 0;
		node = NULL;
	}
}

static void parse_corpus_file() {
	FILE* fin = fopen(CORPUS_PATH, "r");

	if(!fin) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
		return;
	}

	int i = 0, j = 0, success;
	char c, word[WORD_MAX] = { 0 };
	char* pw = word;

	while((c = fgetc(fin)) != EOF) {
		if(isalnum(c)) {
			*pw++ = tolower(c);
		} else if(!isalnum(c) && word[0]) {
			strcpy(context[i][j++], word);
			success = 0;
			root = bst_insert(root, word, &success);
			if(success) {
				pattern_max = input_max = ++output_max;
				hidden_max = HIDDEN_MAX;
			}
			memset(pw = word, 0, sizeof(word));
		} else if(c == '.') {
			++i, j = 0;
		}
	}

	if(fclose(fin) == EOF) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
	}
}

static void allocate_layers() {
	input = (xBit**) calloc(pattern_max, sizeof(xBit*));
	target = (xBit**) calloc(pattern_max, sizeof(xBit*));
	hidden = (double**) calloc(pattern_max, sizeof(double*));
	output = (double**) calloc(pattern_max, sizeof(double*));
	for(p = 0; p < pattern_max; p++) {
		input[p] = (xBit*) calloc(input_max, sizeof(xBit));
		target[p] = (xBit*) calloc(output_max, sizeof(xBit));
		hidden[p] = (double*) calloc(hidden_max, sizeof(double));
		output[p] = (double*) calloc(output_max, sizeof(double));
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
	error_d = (double*) calloc(output_max, sizeof(double));
}

static void free_layers() {
	for(p = 0; p < pattern_max; p++) {
		free(input[p]);
		free(target[p]);
		free(hidden[p]);
		free(output[p]);
	}
	free(input);
	free(target);
	free(hidden);
	free(output);

	for(i = 0; i < input_max; i++) {
		free(weight_ih[i]);
	}
	free(weight_ih);

	for(j = 0; j < hidden_max; j++) {
		free(weight_ho[j]);
	}
	free(weight_ho);

	free(training);
	free(error_d);
}

static void initialize_training() {
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

			for(index = 0; index < input_max && !word[index].on; index++)
				;

			for(k = j - WINDOW_MAX; k <= j + WINDOW_MAX; k++) {
				if(k == j || k < 0 || !context[i][k][0]) {
					continue;
				}

				word = map_get(context[i][k]);

				int pom;

				for(pom = 0; pom < output_max; pom++) {
					target[index][pom].on |= word[pom].on;
				}
			}
		}
	}

	for(p = 0; p < pattern_max; p++) {
		training[p] = p;
	}

	for(i = 0; i < input_max; i++) {
		input[i][i].on = 1;
	}

	fout = fopen(OUTPUT_PATH, "w");
	flog = fopen(LOG_PATH, "w");
}

static void initialize_weights() {
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
}

static void forward_propagate_input_layer() {
	for(j = 0; j < hidden_max; j++) {
		hidden[p][j] = 0.0;

		for(i = 0; i < input_max; i++) {
			hidden[p][j] += input[p][i].on * weight_ih[i][j];
		}
	}
}

static void forward_propagate_hidden_layer() {
	for(k = 0; k < output_max; k++) {
		output[p][k] = 0.0;

		for(j = 0; j < hidden_max; j++) {
			output[p][k] += hidden[p][j] * weight_ho[j][k];
		}
	}
}

static void calculate_error() {
	error = 0.0;
	sum = 0.0;
	count = 0;

	for(k = 0; k < output_max; k++) {
		sum += target[p][k].on * output[p][k];
		count += target[p][k].on;
	}

	error -= sum;
	sum = 0.0;

	for(k = 0; k < output_max; k++) {
		sum += exp(output[p][k]);
	}

	error += count * log(sum);
}

static void normalize_output_layer() {
	max = output[p][0];

	for(k = 0; k < output_max; k++) {
		max = (output[p][k] > max) ? output[p][k] : max;
	}

	sum = 0.0;

	for(k = 0; k < output_max; k++) {
		sum += exp(output[p][k] - max);
	}

	offset = max + log(sum);

	for(k = 0; k < output_max; k++) {
		output[p][k] = exp(output[p][k] - offset);
	}
}

static void calculate_error_derivative() {
	for(k = 0; k < output_max; k++) {
		error_d[k] = output[p][k] - target[p][k].on;
	}
}

static void update_hidden_layer_weights() {
	for(j = 0; j < hidden_max; j++) {
		for(k = 0; k < output_max; k++) {
			weight_ho[j][k] -= hidden[p][j] * error_d[k] * LEARNING_RATE;
		}
	}
}

static void update_input_layer_weights() {
	for(i = 0; i < input_max; i++) {
		for(j = 0; j < hidden_max; j++) {
			for(k = 0; k < output_max; k++) {
				weight_ih[i][j] -= input[p][i].on * weight_ho[j][k] * error_d[k] * LEARNING_RATE;
			}
		}
	}
}

static void log_epoch() {
	if(epoch % LOG_PERIOD) {
		return;
	}

	fprintf(LOG_FILE, "%cEpoch\t%d\n", epoch ? '\n' : 0, epoch + 1);
	fprintf(LOG_FILE, "Error\t%lf\n", error);
	fprintf(LOG_FILE, "Took\t%lf sec\n", (double) (elapsed = clock() - elapsed) / CLOCKS_PER_SEC);
	fprintf(LOG_FILE, "Input\tTarget\t\tOutput\t\tError\n");

	for(i = 0; i < input_max; i++) {
		fprintf(LOG_FILE, "%d\t%d\t\t%lf\t%lf\n", input[p][i].on, target[p][i].on, output[p][i], error_d[i]);
	}
}

static void save_output() {
	for(p = 0; p < pattern_max; p++) {
		if(!p) {
			fprintf(fout, "%2d:", p);

			for(k = 0; k < output_max; k++) {
				fprintf(fout, "%3d", k);
			}

			fprintf(fout, "\n");
		}

		fprintf(fout, "%2d:", p);

		for(k = 0; k < output_max; k++) {
			fprintf(fout, "%3d", target[p][k].on);
		}

		fprintf(fout, "\n");
	}

	fprintf(fout, "\n");

	for(p = 0; p < pattern_max; p++) {
		if(!p) {
			fprintf(fout, "%2d:", p);

			for(k = 0; k < output_max; k++) {
				fprintf(fout, "%7d", k);
			}

			fprintf(fout, "\n");
		}

		fprintf(fout, "%2d:", p);

		for(k = 0; k < output_max; k++) {
			fprintf(fout, "%7.3lf", output[p][k]);
		}

		fprintf(fout, "\n");
	}
}

static void finish_training() {
	bst_free(root);
	free_layers();
	fclose(fout);
	fclose(flog);
}

static int cmp_double(const void* a, const void* b) {
	if(*(double*) a > *(double*) b) {
		return 1;
	}else if(*(double*) a < *(double*) b) {
		return -1;
	}else {
		return 0;
	}
}

void start_training() {
	srand(time(0));

	initialize_training();
	initialize_weights();

	elapsed = clock();

	for(epoch = 0; epoch < EPOCH_MAX; epoch++) {
		printf("Epoch %d / %d\n", epoch + 1, EPOCH_MAX);

		initialize_epoch();

		for(p1 = 0; p1 < pattern_max; p1++) {
			p = training[p1];

			forward_propagate_input_layer();
			forward_propagate_hidden_layer();
			calculate_error();
			normalize_output_layer();
			calculate_error_derivative();
			update_hidden_layer_weights();
			update_input_layer_weights();
		}

		if(LOG_EPOCH) {
			log_epoch();
		}
	}

	save_output();
	finish_training();
}

// TODO Save weights in file
// TODO Load weights from file

void get_predictions(const char* word, int count) {
	xBit* onehot = map_get(word);
	
	// TODO Value of p?
	
	double* pred[output_max];

	for(k = 0; k < output_max; k++) {
		pred = output[p][k];
	}

	// TODO Get pred_word from output

	qsort(pred, output_max, sizeof(double), cmp_double);
	
	for(k = 0; k < count; k++) {
		printf("#%d\t%s\t%lf\n", k, pred_word, pred[k]);
	}
}
