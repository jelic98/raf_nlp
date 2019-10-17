// http://www.cs.bham.ac.uk/~jxb/INC/nn.html

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>

#define PATTER_MAX 4
#define INPUT_MAX 2
#define HIDDEN_MAX 2
#define OUTPUT_MAX 1

#define LEARNING_RATE 0.5
#define MOMENTUM 0.9
#define INITIAL_WEIGHT_MAX 0.5

#define EPOCH_MAX 10000
#define ERROR_MAX 0.0004

#define LOG_PERIOD 100
#define OUT stdout

#define random() ((double)rand() / ((double) RAND_MAX + 1))

static int i, j, k, p, np, op, training[PATTER_MAX + 1], epoch;

static double error;

static int num_pattern = PATTER_MAX,
	num_input = INPUT_MAX,
	num_hidden = HIDDEN_MAX,
	num_output = OUTPUT_MAX;

static double input[PATTER_MAX + 1][INPUT_MAX + 1] = {
	{0, 0, 0},
	{0, 0, 0},
	{0, 1, 0},
	{0, 0, 1},
	{0, 1, 1}
};

static double target[PATTER_MAX + 1][OUTPUT_MAX + 1] = {
	{0, 0},
	{0, 0},
	{0, 1},
	{0, 1},
	{0, 0}
};

static double sum_h[PATTER_MAX + 1][HIDDEN_MAX + 1],
	weight_ih[INPUT_MAX + 1][HIDDEN_MAX + 1],
	hidden[PATTER_MAX + 1][HIDDEN_MAX + 1];

static double sum_o[PATTER_MAX + 1][OUTPUT_MAX + 1],
	weight_ho[HIDDEN_MAX + 1][OUTPUT_MAX + 1],
	output[PATTER_MAX + 1][OUTPUT_MAX + 1];

static double delta_o[OUTPUT_MAX + 1],
	sum_dow[HIDDEN_MAX + 1],
	delta_h[HIDDEN_MAX + 1];

static double delta_weight_ih[INPUT_MAX + 1][HIDDEN_MAX + 1],
	delta_weight_ho[HIDDEN_MAX + 1][OUTPUT_MAX + 1];

static void initialize_weights() {
	// Initialize weights from input to hidden layer
	for(j = 1; j <= num_hidden; j++) {
		for(i = 0; i <= num_input; i++) {
			delta_weight_ih[i][j] = 0.0;
			weight_ih[i][j] = 2.0 * (random() - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}

	// Initialize weights from hidden to output layer
	for(k = 1; k <= num_output; k++) {
		for(j = 0; j <= num_hidden; j++) {
			delta_weight_ho[j][k] = 0.0;
			weight_ho[j][k] = 2.0 * (random() - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}
}

static void initialize_training() {
	// Randomize order inside training set
	for(p = 1; p <= num_pattern; p++) {
		training[p] = p;
	}

	// Randomize order inside training set
	for(p = 1; p <= num_pattern; p++) {
		np = p + random() * (num_pattern + 1 - p);
		op = training[p];
		training[p] = training[np];
		training[np] = op;
	}
}

static void update_weights() {
	// Update weights from input to hidden layer
	for(j = 1; j <= num_hidden; j++) {
		delta_weight_ih[0][j] = LEARNING_RATE * delta_h[j] + MOMENTUM * delta_weight_ih[0][j];
		weight_ih[0][j] += delta_weight_ih[0][j];

		for(i = 1; i <= num_input; i++) {
			delta_weight_ih[i][j] = LEARNING_RATE * input[p][i] * delta_h[j] + MOMENTUM * delta_weight_ih[i][j];
			weight_ih[i][j] += delta_weight_ih[i][j];
		}
	}

	// Update weights from hidden to output layer
	for(k = 1; k <= num_output; k++) {
		delta_weight_ho[0][k] = LEARNING_RATE * delta_o[k] + MOMENTUM * delta_weight_ho[0][k];
		weight_ho[0][k] += delta_weight_ho[0][k];

		for(j = 1; j <= num_hidden; j++) {
			delta_weight_ho[j][k] = LEARNING_RATE * hidden[p][j] * delta_o[k] + MOMENTUM * delta_weight_ho[j][k];
			weight_ho[j][k] += delta_weight_ho[j][k];
		}
	}
}

static void log_epoch() {
	if(epoch % LOG_PERIOD == 0) {
		fprintf(OUT, "Epoch %-5d\t:\tError = %f\n", epoch, error);
	}
} 

static void log_network() {
	fprintf(OUT, "\nNETWORK DATA - EPOCH %d\n\n#\t", epoch);
  
	for(i = 1; i <= num_input; i++) {
    	fprintf(OUT, "Input %-4d\t", i);
	}
	
	for(k = 1; k <= num_output; k++) {
    	fprintf(OUT, "Target %-4d\tOutput %-4d\t", k, k);
	}
	
	for(p = 1; p <= num_pattern; p++) {
		fprintf(OUT, "\n%d\t", p);
		
		for(i = 1; i <= num_input; i++) {
			fprintf(OUT, "%f\t", input[p][i]);
		}
		
		for(k = 1; k <= num_output; k++) {
			fprintf(OUT, "%f\t%f\t", target[p][k], output[p][k]);
		}
	}
}

static void back_propagate_error() {
	for(j = 1; j <= num_hidden; j++) {
		sum_dow[j] = 0.0;
		
		for(k = 1; k <= num_output; k++) {
			sum_dow[j] += weight_ho[j][k] * delta_o[k];
		}
		
		delta_h[j] = sum_dow[j] * hidden[p][j] * (1.0 - hidden[p][j]);
	}	
}

static void compute_output_layer() {
	// Compute output layer activations and errors
	for(k = 1; k <= num_output; k++) {
		// Add bias
		sum_o[p][k] = weight_ho[0][k];

		// Sum outputs from hidden layer
		for(j = 1; j <= num_hidden; j++) {
			sum_o[p][k] += hidden[p][j] * weight_ho[j][k];
		}

		// Compute sigmoid function
		output[p][k] = 1.0 / (1.0 + exp(-sum_o[p][k]));

		// Comput sum squared error
		error += 0.5 * (target[p][k] - output[p][k]) * (target[p][k] - output[p][k]);

		// Compute output layer delta
		delta_o[k] = (target[p][k] - output[p][k]) * output[p][k] * (1.0 - output[p][k]);
	}
}

static void compute_hidden_layer() {
	// Compute hidden layer activations
	for(j = 1; j <= num_hidden; j++) {
		// Add bias
		sum_h[p][j] = weight_ih[0][j];
			
		// Sum outputs from input layer
		for(i = 1; i <= num_input; i++) {
			sum_h[p][j] += input[p][i] * weight_ih[i][j];
		}

		// Compute sigmoid function
		hidden[p][j] = 1.0 / (1.0 + exp(-sum_h[p][j]));
	}
}

static void start_training() {
	initialize_weights();
	
	for(epoch = 0; epoch < EPOCH_MAX; epoch++) {
		error = 0.0;

		initialize_training();

		for(np = 1; np <= num_pattern; np++) {
			p = training[np];

			compute_hidden_layer();
			compute_output_layer();
			back_propagate_error();
			update_weights();
		}
		
		log_epoch();
		
		if(error < ERROR_MAX) {
			break;
		}
	}
	
	log_network();
}

int main() {
	srand(time(0));	
	
	start_training();
	
	return 0;
}
