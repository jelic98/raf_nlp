#include "nn.h"

dt_char arg_train[PATH_CHARACTER_MAX] = {0};
dt_char arg_test[PATH_CHARACTER_MAX] = {0};
dt_char arg_stop[PATH_CHARACTER_MAX] = {0};
dt_char arg_out[PATH_CHARACTER_MAX] = {0};

int main(int argc, char* argv[]) {	
	if(argc == 4) {
		strcpy(arg_train, argv[1]);
		strcpy(arg_test, argv[2]);
		strcpy(arg_stop, argv[3]);
	}else {
		printf(ERROR_CMDARGS);
		exit(1);
	}
	
	nn_start();
	training_run();
	weights_save();
	nn_finish();

	return 0;
}
