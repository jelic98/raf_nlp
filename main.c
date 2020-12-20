#include "nn.h"

int main(int argc, char* argv[]) {
	nn_start();
	
	if(!argv[1] || strcmp(argv[1], "--vocab-only")) {
		training_run();
		weights_save();
	}
	
	nn_finish();

	return 0;
}
