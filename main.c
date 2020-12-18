#include "nn.h"

int main(int argc, char* argv[]) {
	nn_start();
	training_run();
	weights_save();
	nn_finish();

	return 0;
}
