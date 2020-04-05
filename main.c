#include "lib.h"
#include "nn.h"
#include "stmr.h"

// TODO [PARSER] multithreading

// TODO [MODEL] negative sampling
// TODO [MODEL] dropout rate for w_ih
// TODO [MODEL] normal distribution for weight initialization
// TODO [MODEL] run training in batches

// TODO [MISC] tracking elapsed time for testing and parsing
// TODO [MISC] all functions should be void

int main() {
	nn_start();
	training_run();
	test_run();
	nn_finish();

	return 0;
}
