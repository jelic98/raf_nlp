#include "nn.h"

// https://stats.stackexchange.com/questions/244616/how-does-negative-sampling-work-in-word2vec

// TODO [MODEL] negative sampling
// TODO [MODEL] dropout rate for w_ih
// TODO [MODEL] normal distribution for weight initialization
// TODO [MODEL] run training in batches

// TODO [PARSER] multithreading
// TODO [MISC] all functions should be void
// TODO [MISC] match every malloc with free

int main() {
	nn_start();
	training_run();
	test_run();
	nn_finish();

	return 0;
}
