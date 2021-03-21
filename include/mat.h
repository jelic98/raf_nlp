#ifndef H_MAT_INCLUDE
#define H_MAT_INCLUDE

#include "lib.h"

void vector_normalize(dt_float*, dt_int);
void vector_softmax(dt_float*, dt_int);
void vector_distance(dt_float*, dt_float*, dt_int, dt_float*);
dt_int cmp_dist(const void*, const void*);
dt_int cmp_prob(const void*, const void*);
dt_float sigmoid(dt_float);
dt_float xavier(dt_int, dt_int);

#ifdef H_MAT_IMPLEMENT
// Normalize vector to length of 1
void vector_normalize(dt_float* vector, dt_int size) {
	dt_float sum;
	dt_int q;

	for(sum = q = 0; q < size; q++) {
		sum += vector[q] * vector[q];
	}

	for(sum = sqrt(sum), q = 0; q < size; q++) {
		vector[q] /= sum;
	}
}

#ifdef FLAG_TEST_CONTEXT
// Apply softmax function to every vector component
void vector_softmax(dt_float* vector, dt_int size) {
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

// Calculate Euclidean distance between two vectors
void vector_distance(dt_float* v1, dt_float* v2, dt_int size, dt_float* dist) {
	dt_int k;
	dt_float sum;

	dt_float sum1, sum2;

	for(sum = sum1 = sum2 = k = 0; k < size; k++) {
		sum += v1[k] * v2[k];
		sum1 += v1[k] * v1[k];
		sum2 += v2[k] * v2[k];
	}

	sum1 = sqrt(sum1);
	sum2 = sqrt(sum2);

	*dist = sum / (sum1 * sum2);
}

#ifdef FLAG_TEST_SIMILARITY
// Compare two predicted words by their distance to target word
dt_int cmp_dist(const void* a, const void* b) {
	dt_float diff = (*(xWord**) b)->dist - (*(xWord**) a)->dist;
	return diff > 0 ? 1 : diff < 0 ? -1 : 0;
}
#endif

#ifdef FLAG_TEST_CONTEXT
// Compare two predicted words by their probability at output layer
dt_int cmp_prob(const void* a, const void* b) {
	dt_float diff = (*(xWord**) b)->prob - (*(xWord**) a)->prob;
	return diff > 0 ? 1 : diff < 0 ? -1 : 0;
}
#endif

// Calculate sigmoid function
dt_float sigmoid(dt_float x) {
	return 1.0 / (1.0 + exp(-x));
}

// Calculate normalized Xavier weight initiazation value
dt_float xavier(dt_int n, dt_int m) {
	return random_unif(-1.0, 1.0) * sqrt(6.0 / (n + m));
}
#endif
#endif
