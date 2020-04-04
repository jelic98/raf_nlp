#include "include/main.h"
#include "include/nn.h"

// TODO [PARSER] word stemmer
// TODO [PARSER] multithreading

// TODO [MODEL] negative sampling
// TODO [MODEL] dropout rate for w_ih
// TODO [MODEL] normal distribution for weight initialization

// TODO [MISC] detailed logging with timestamps
// TODO [MISC] all functions should be void
// TODO [MISC] print memory allocation errors
// TODO [MISC] reset corpus and other pointers to null after release

int main() {
	nn_start();
	training_run();
	//weights_save();
	//weights_load();
	test_run();
	nn_finish();

	return 0;
}
