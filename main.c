#include "include/main.h"
#include "include/nn.h"

// TODO [PARSER] word stemmer
// TODO [PARSER] multithreading

// TODO [MODEL] fix loss function
// TODO [MODEL] negative sampling
// TODO [MODEL] dropout rate for w_ih
// TODO [MODEL] normal distribution for weight initialization
// TODO [MODEL] run training in batches

// TODO [MISC] detailed logging with timestamps
// TODO [MISC] all functions should be void
// TODO [MISC] print memory allocation errors

int main() {
	nn_start();
	training_run();
	//weights_save();
	//weights_load();
	test_run();
	nn_finish();

	return 0;
}
