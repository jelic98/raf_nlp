#include "include/nn.h"

static clock_t elapsed; 

// Input layer index
static int i;

// Hidden layer index
static int j;

// Output layer index
static int k;

// Pattern indices
static int p, p1, p2;

static int training[PATTERN_MAX], epoch;

// Softmax variables
static double max, sum, offset;

static double delta, loss;

static xBit input[PATTERN_MAX][INPUT_MAX],
	target[PATTERN_MAX][OUTPUT_MAX];

static double weight_ih[INPUT_MAX][HIDDEN_MAX],
	hidden[PATTERN_MAX][HIDDEN_MAX];

static double weight_ho[HIDDEN_MAX][OUTPUT_MAX],
	output[PATTERN_MAX][OUTPUT_MAX];

static double error[OUTPUT_MAX];

static xWord* root = NULL;
static char context[SENTENCE_MAX][WORD_MAX][CHARACTER_MAX];
static xBit onehot[SENTENCE_MAX * WORD_MAX][SENTENCE_MAX * WORD_MAX];

static xBit* map_get(const char* word) {
	unsigned int hash = 0, c;

	for(size_t i = 0; word[i]; i++) {
		c = (unsigned char) word[i];
		hash = (hash << 3) + (hash >> (sizeof(hash) * CHAR_BIT - 3)) + c;
	}

	return onehot[hash % (SENTENCE_MAX * WORD_MAX)];
}

static xWord* bst_insert(xWord* node, const char* word) {
	if(!node) {
		node = (xWord*) malloc(sizeof(xWord));
		strcpy(node->word, word);
		node->left = node->right = NULL;
		node->count = 1;
		return node;
	}

	int cmp = strcmp(word, node->word);
	
	if(cmp < 0) {
		node->left = bst_insert(node->left, word);
	}else if(cmp > 0) {
		node->right = bst_insert(node->right, word);
	}else {
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

static void bst_clear(xWord* node) {
	if(node) {
		bst_clear(node->left);
		bst_clear(node->right);
		node->left = NULL;
		node->right = NULL;
		free(node);
		memset(node->word, 0, sizeof(node->word));
		node->count = 0;
		node = NULL;
	}
}

static void parse_corpus_file() {
	FILE* fin = fopen(CORPUS_FILE, "r");

	if(!fin) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
		return;
	}

	int i = 0, j = 0;
	char c, word[WORD_MAX] = {0};
	char* pw = word;

	while((c = fgetc(fin)) != EOF) {
		if(isalnum(c)) {
			*pw++ = tolower(c);
		}else if(!isalnum(c) && word[0]) {
			strcpy(context[i][j++], word);
			root = bst_insert(root, word);
			memset(pw = word, 0, sizeof(word));
		}else if(c == '.') {
			++i, j = 0;
		}	
	}

	if(fclose(fin) == EOF) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
	}
}

static void initialize_training() {
	parse_corpus_file();
	
	int index = 0;

	// Convert binary search tree to map
	bst_to_map(root, &index);

	// Initialize input layer and target outputs
	for(i = 0; i < SENTENCE_MAX; i++) {
		for(j = 0; j < WORD_MAX; j++) {
			if(!context[i][j][0]) {
				continue;
			}
			
			xBit* word = map_get(context[i][k]);

			for(index = 0; index < INPUT_MAX && !word[index].on; index++);
			
			input[index][index].on = 1;

			for(k = j - WINDOW_MAX; k <= j + WINDOW_MAX; k++) {
				if(k == j || k < 0 || !context[i][k][0]) {
					continue;
				}
			
				word = map_get(context[i][k]);

				int pom;

				for(pom = 0; pom < INPUT_MAX; pom++) {
					target[index][pom].on |= word[pom].on;
				}
			}
		}	
	}

	// Initialize training set
	for(p = 0; p < PATTERN_MAX; p++) {
		training[p] = p;
	}
}

static void initialize_weights() {
	// Initialize weights from input to hidden layer
	for(j = 0; j < HIDDEN_MAX; j++) {
		for(i = 0; i < INPUT_MAX; i++) {
			weight_ih[i][j] = 2.0 * (random() - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}

	// Initialize weights from hidden to output layer
	for(k = 0; k < OUTPUT_MAX; k++) {
		for(j = 0; j < HIDDEN_MAX; j++) {
			weight_ho[j][k] = 2.0 * (random() - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}
}

static void initialize_epoch() {
	loss = 0.0;

	// Randomize order inside training set
	for(p = 0; p < PATTERN_MAX; p++) {
		p1 = p + random() * (PATTERN_MAX - p);
		
		p2 = training[p];
		training[p] = training[p1];
		training[p1] = p2;
	}
}

static void forward_propagate_input_layer() {
	// Compute hidden layer
	for(j = 0; j < HIDDEN_MAX; j++) {
		hidden[p][j] = 0.0;

		// Sum outputs from input layer
		for(i = 0; i < INPUT_MAX; i++) {
			hidden[p][j] += input[p][i].on * weight_ih[i][j];
		}
	}
}

static void forward_propagate_hidden_layer() {
	// Compute output layer
	for(k = 0; k < OUTPUT_MAX; k++) {
		output[p][k] = 0.0;
		
		// Sum outputs from hidden layer
		for(j = 0; j < HIDDEN_MAX; j++) {
			output[p][k] += hidden[p][j] * weight_ho[j][k];
		}
	}
}

static void normalize_output_layer() {
	// Run softmax function through output layer to normalize it into probability distribution
	max = output[p][k];

	for(k = 0; k < OUTPUT_MAX; k++) {
		if(output[p][k] > max) {
			max = output[p][k];
		}
	}

	sum = 0.0;
	
	for(k = 0; k < OUTPUT_MAX; k++) {
		sum += expf(output[p][k] - max);
	}
	
	offset = max + log(sum);

	for(k = 0; k < OUTPUT_MAX; k++) {
		output[p][k] = exp(output[p][k] - offset);
	}
}

static void calculate_error() {
	// Calculate error based on targets and probabilities at output layer
	for(k = 0; k < OUTPUT_MAX; k++) {
		error[k] = output[p][k] - target[p][k].on;
	}
}

static void calculate_loss() {
// Part 1: -ve sum of all the output +
// Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
// Note: word.index(1) returns the index in the context word vector with value 1
// Note: u[word.index(1)] returns the value of the output layer before softmax
// self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
/* sum = 0; */

/* for(k = 0; k < OUTPUT_MAX; k++) { */
/* sum += output[p][k]; */
/* } */
}

static void update_hidden_layer_weights() {
	// Update weights from hidden to output layer
	for(k = 0; k < OUTPUT_MAX; k++) {
		// Apply delta rule
		for(j = 0; j < HIDDEN_MAX; j++) {
			weight_ho[j][k] -= LEARNING_RATE * hidden[p][j] * error[k];
		}
	}
}

static void update_input_layer_weights() {
	// Update weights from input to hidden layer
	for(i = 0; i < INPUT_MAX; i++) {
		// Apply delta rule
		for(j = 0; j < HIDDEN_MAX; j++) {
			delta = 0.0;

			for(k = 0; k < OUTPUT_MAX; k++) {
				delta += hidden[p][j] * error[k];
			}
	
			weight_ih[i][j] -= LEARNING_RATE * delta;
		}
	}
}

static void log_network() {
	if(epoch % LOG_PERIOD ) {
		return;
	}

	fprintf(LOG_FILE, "%cEpoch\t%d\n", epoch ? '\n' : 0, epoch);
	fprintf(LOG_FILE, "Loss\t%lf\n", loss);
	fprintf(LOG_FILE, "Input\tOutput\t\tTarget\n");
		
	for(i = 0; i < INPUT_MAX; i++) {
		fprintf(LOG_FILE, "%d\t%lf\t%d\n", input[p][i].on, output[p][i], target[p][i].on);
	}

	fprintf(LOG_FILE, "\nElapsed time: %f sec\n", (double) elapsed / CLOCKS_PER_SEC);
}

void start_training() {
	srand(time(0));

	initialize_training();	
	initialize_weights();	

	elapsed = clock();

	for(epoch = 0; epoch < EPOCH_MAX; epoch++) {
		initialize_epoch();

		for(p1 = 0; p1 < PATTERN_MAX; p1++) {
			p = training[p1];

			forward_propagate_input_layer();
			forward_propagate_hidden_layer();
			normalize_output_layer();
			calculate_error();
			calculate_loss();
			update_hidden_layer_weights();
			update_input_layer_weights();
		}
		
//		log_network();
		
		if(loss < LOSS_MAX) {
			break;
		}
	}
    
	elapsed = clock() - elapsed; 

	bst_clear(root);
}
