#include "nn.h"

int main() {
	nn_start();
	training_run();
	weights_save();
	//testing_run();
	//nn_finish();

	return 0;
}
