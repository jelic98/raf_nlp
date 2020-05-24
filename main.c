#include "nn.h"

int main() {
	nn_start();
	training_run();
	testing_run();
	nn_finish();

	return 0;
}
