#include "include/main.h"

#define H_TEST_INCLUDE
#include "include/nn.h"

int main() {
	//load_weights();
	start_training();
	save_weights();

	char test[TEST_MAX][CHARACTER_MAX] = { TEST_CASES };
	int i, result, tries_sum = 0;

	for(i = 0; i < TEST_MAX && test[i][0]; i++) {
		get_predictions(test[i], 5, &result);
		tries_sum += result;
	}

	printf("Precision: %.1lf%%\n", 100.0 * tries_sum / TEST_MAX);

	finish_training();

	return 0;
}
