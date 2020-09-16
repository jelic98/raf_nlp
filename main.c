#include "nn.h"

int main() {
	nn_start();
	training_run();
	weights_save();
	sentences_encode();
	sentences_similarity();
	testing_run();
	nn_finish();

	return 0;
}
