#include "nn.h"

int main() {
	nn_start();
	training_run();
	test_run();
	nn_finish();

	return 0;
}
