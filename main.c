#include "nn.h"

int main() {
	nn_start();
	training_run();
	weights_save();
	testing_run();
	sentences_encode();
	nn_finish();

	return 0;
}
