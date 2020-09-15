#include "nn.h"

int main() {
	nn_start();
	training_run();
	weights_save();
	sentences_encode();
	testing_run();
	nn_finish();

	return 0;
}
