#include "include/nn.h"

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

static int training[PATTERN_MAX], epoch;

// Softmax variables
static double max, sum, offset;

static double delta, loss;

static double input[PATTERN_MAX][INPUT_MAX] = {
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1}
};

static double target[PATTERN_MAX][OUTPUT_MAX] = {
	{0},
	{1},
	{1},
	{0}
};

static double weight_ih[INPUT_MAX][HIDDEN_MAX],
	hidden[PATTERN_MAX][HIDDEN_MAX];

static double weight_ho[HIDDEN_MAX][OUTPUT_MAX],
	output[PATTERN_MAX][OUTPUT_MAX];

static double error[OUTPUT_MAX];

static void initialize_training() {
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
			hidden[p][j] += input[p][i] * weight_ih[i][j];
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
		error[k] = output[p][k] - target[p][k];
	}
}

static void calculate_loss() {
// Part 1: -ve sum of all the output +
// Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
// Note: word.index(1) returns the index in the context word vector with value 1
// Note: u[word.index(1)] returns the value of the output layer before softmax
// self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
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

static void log_epoch() {
	if(epoch % LOG_PERIOD == 0) {
		fprintf(LOG_FILE, "Epoch %d\t:\tLoss = %f\n", epoch, loss);
	}
} 

static void log_network() {
	fprintf(LOG_FILE, "\nNetwork data (Epoch %d)\n\n#\t", epoch);
  
	for(i = 0; i < INPUT_MAX; i++) {
		fprintf(LOG_FILE, "Input %-4d\t", i);
	}
	
	for(k = 0; k < OUTPUT_MAX; k++) {
		fprintf(LOG_FILE, "Target %-4d\tOutput %-4d\t", k, k);
	}
	
	for(p = 0; p < PATTERN_MAX; p++) {
		fprintf(LOG_FILE, "\n%d\t", p);
		
		for(i = 0; i < INPUT_MAX; i++) {
			fprintf(LOG_FILE, "%f\t", input[p][i]);
		}
		
		for(k = 0; k < OUTPUT_MAX; k++) {
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
		
		log_epoch();
		
		if(loss < LOSS_MAX) {
			break;
		}
	}
    
	elapsed = clock() - elapsed; 
	
	log_network();
}
