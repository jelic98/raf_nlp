#include "lib.h"
#include "nn.h"
#include "stmr.h"

// TODO does weight init get called if weights are loaded?
// TODO [BACKUP] save weights after each epoch

// TODO [PARSER] multithreading

// TODO [MODEL] negative sampling
// TODO [MODEL] dropout rate for w_ih
// TODO [MODEL] normal distribution for weight initialization
// TODO [MODEL] run training in batches

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
