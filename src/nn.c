#include "include/nn.h"

// TODO Disable activation function and only use matrix dot product
// TODO Use softmax function to compute probabilities
// TODO Optimize calculations by reusing calculated expessions

static clock_t elapsed; 

// Input layer index
static int i;

// Hidden layer index
static int j;

// Output layer index
static int k;

// Pattern indices
static int p, p1, p2;

static int training[PATTERN_MAX + 1], epoch;

static double sum_squares, error;

static double input[PATTERN_MAX + 1][INPUT_MAX + 1] = {
	{0, 0, 0},
	{0, 0, 0},
	{0, 1, 0},
	{0, 0, 1},
	{0, 1, 1}
};

static double target[PATTERN_MAX + 1][OUTPUT_MAX + 1] = {
	{0, 0},
	{0, 0},
	{0, 1},
	{0, 1},
	{0, 0}
};

static double sum_hidden[PATTERN_MAX + 1][HIDDEN_MAX + 1],
	weight_ih[INPUT_MAX + 1][HIDDEN_MAX + 1],
	hidden[PATTERN_MAX + 1][HIDDEN_MAX + 1];

static double sum_output[PATTERN_MAX + 1][OUTPUT_MAX + 1],
	weight_ho[HIDDEN_MAX + 1][OUTPUT_MAX + 1],
	output[PATTERN_MAX + 1][OUTPUT_MAX + 1];

static double sum_delta[HIDDEN_MAX + 1],
	delta_h[HIDDEN_MAX + 1],
	delta_o[OUTPUT_MAX + 1];

static double delta_weight_ih[INPUT_MAX + 1][HIDDEN_MAX + 1],
	delta_weight_ho[HIDDEN_MAX + 1][OUTPUT_MAX + 1];

static void initialize_training() {
	// Initialize training set
	for(p = 1; p <= PATTERN_MAX; p++) {
		training[p] = p;
	}
}

static void initialize_weights() {
	// Initialize weights from input to hidden layer
	for(j = 1; j <= HIDDEN_MAX; j++) {
		for(i = 0; i <= INPUT_MAX; i++) {
			delta_weight_ih[i][j] = 0.0;
			weight_ih[i][j] = 2.0 * (random() - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}

	// Initialize weights from hidden to output layer
	for(k = 1; k <= OUTPUT_MAX; k++) {
		for(j = 0; j <= HIDDEN_MAX; j++) {
			delta_weight_ho[j][k] = 0.0;
			weight_ho[j][k] = 2.0 * (random() - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}
}

static void initialize_epoch() {
	error = 0.0;
	
	// Randomize order inside training set
	for(p = 1; p <= PATTERN_MAX; p++) {
		p1 = p + random() * (PATTERN_MAX + 1 - p);
		
		p2 = training[p];
		training[p] = training[p1];
		training[p1] = p2;
	}
}

static void forward_propagate_input_layer() {
	// Compute hidden layer activations
	for(j = 1; j <= HIDDEN_MAX; j++) {
		// Add bias
		sum_hidden[p][j] = weight_ih[0][j];
			
		// Sum outputs from input layer
		for(i = 1; i <= INPUT_MAX; i++) {
			sum_hidden[p][j] += input[p][i] * weight_ih[i][j];
		}

		// Compute sigmoid function
		hidden[p][j] = 1.0 / (1.0 + exp(-sum_hidden[p][j]));
	}
}

static void forward_propagate_hidden_layer() {
	sum_squares = 0.0;

	// Compute output layer activations and deltas
	for(k = 1; k <= OUTPUT_MAX; k++) {
		// Add bias
		sum_output[p][k] = weight_ho[0][k];

		// Sum outputs from hidden layer
		for(j = 1; j <= HIDDEN_MAX; j++) {
			sum_output[p][k] += hidden[p][j] * weight_ho[j][k];
		}

		// Compute sigmoid function
		output[p][k] = 1.0 / (1.0 + exp(-sum_output[p][k]));
		
		// Compute output layer delta
		delta_o[k] = (target[p][k] - output[p][k]) * output[p][k] * (1.0 - output[p][k]);

		// Compute sum of squared Euclidean distance
		sum_squares += pow((target[p][k] - output[p][k]), 2);
	}

	// Compute mean squared error
	error = 0.5 * sum_squares / OUTPUT_MAX;
}

static void back_propagate_error() {
	// Compute hidden layer deltas
	for(j = 1; j <= HIDDEN_MAX; j++) {
		sum_delta[j] = 0.0;
		
		// Sum products of output layer deltas and weights from hidden to output layer
		for(k = 1; k <= OUTPUT_MAX; k++) {
			sum_delta[j] += delta_o[k] * weight_ho[j][k];
		}
		
		// Compute hidden layer delta
		delta_h[j] = sum_delta[j] * hidden[p][j] * (1.0 - hidden[p][j]);
	}	
}

static void update_weights() {
	// Update weights from input to hidden layer
	for(j = 1; j <= HIDDEN_MAX; j++) {
		// Apply delta rule on bias
		delta_weight_ih[0][j] = LEARNING_RATE * delta_h[j] + MOMENTUM_RATE * delta_weight_ih[0][j];
		weight_ih[0][j] += delta_weight_ih[0][j];

		// Apply delta rule on neurons
		for(i = 1; i <= INPUT_MAX; i++) {
			delta_weight_ih[i][j] = LEARNING_RATE * input[p][i] * delta_h[j] + MOMENTUM_RATE * delta_weight_ih[i][j];
			weight_ih[i][j] += delta_weight_ih[i][j];
		}
	}

	// Update weights from hidden to output layer
	for(k = 1; k <= OUTPUT_MAX; k++) {
		// Apply delta rule on bias
		delta_weight_ho[0][k] = LEARNING_RATE * delta_o[k] + MOMENTUM_RATE * delta_weight_ho[0][k];
		weight_ho[0][k] += delta_weight_ho[0][k];

		// Apply delta rule on neurons
		for(j = 1; j <= HIDDEN_MAX; j++) {
			delta_weight_ho[j][k] = LEARNING_RATE * hidden[p][j] * delta_o[k] + MOMENTUM_RATE * delta_weight_ho[j][k];
			weight_ho[j][k] += delta_weight_ho[j][k];
		}
	}
}

static void log_epoch() {
	if(epoch % LOG_PERIOD == 0) {
		fprintf(LOG_FILE, "Epoch %-*d\t:\tError = %f\n", EPOCH_MAX_DIGITS, epoch, error);
	}
} 

static void log_network() {
	fprintf(LOG_FILE, "\nNetwork data (Epoch %d)\n\n#\t", epoch);
  
	for(i = 1; i <= INPUT_MAX; i++) {
		fprintf(LOG_FILE, "Input %-4d\t", i);
	}
	
	for(k = 1; k <= OUTPUT_MAX; k++) {
		fprintf(LOG_FILE, "Target %-4d\tOutput %-4d\t", k, k);
	}
	
	for(p = 1; p <= PATTERN_MAX; p++) {
		fprintf(LOG_FILE, "\n%d\t", p);
		
		for(i = 1; i <= INPUT_MAX; i++) {
			fprintf(LOG_FILE, "%f\t", input[p][i]);
		}
		
		for(k = 1; k <= OUTPUT_MAX; k++) {
			fprintf(LOG_FILE, "%f\t%f\t", target[p][k], output[p][k]);
		}
	}

	fprintf(LOG_FILE, "\n\nElapsed time: %f sec\n", (double) elapsed / CLOCKS_PER_SEC);
}

void start_training() {
	srand(time(0));

	elapsed = clock();

	initialize_training();	
	initialize_weights();	
	
	for(epoch = 0; epoch < EPOCH_MAX; epoch++) {
		initialize_epoch();

		for(p1 = 1; p1 <= PATTERN_MAX; p1++) {
			p = training[p1];

			forward_propagate_input_layer();
			forward_propagate_hidden_layer();
			back_propagate_error();
			update_weights();
		}
		
		log_epoch();
		
		if(error < ERROR_MAX) {
			break;
		}
	}
    
	elapsed = clock() - elapsed; 
	
	log_network();
}
