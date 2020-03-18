#include "include/main.h"

#define H_TEST_INCLUDE
#include "include/nn.h"

int main() {
	nn_start();
	
	training_run();
	//weights_save();
	//weights_load();

	char test[TEST_MAX][100] = { TEST_CASES };
	int i, result = 0, tries_sum = 0;

	for(i = 0; i < TEST_MAX && test[i][0]; i++) {
		test_run(test[i], 5, &result);
		tries_sum += result;
	}

	printf("\nPrecision: %.1lf%%\n", 100.0 * tries_sum / TEST_MAX);
	
	nn_finish();

	return 0;
}
