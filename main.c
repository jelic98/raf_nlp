#include "include/main.h"
#include "include/nn.h"

// TODO ADD training from file not from context matrix
// TODO FIX negative sampling
// TODO ADD word stemmer
// TODO ADD multithreading
// TODO ADD dropout rate for w_ih
// TODO ADD normal distribution for weight initialization
// TODO ADD corpus cleaning
// TODO ADD detailed logging with timestamps
// TODO ADD all functions should be void

int main() {
	nn_start();
	training_run();
	//weights_save();
	//weights_load();
	//test_run();
	nn_finish();

	return 0;
}
