#include "nn.h"

dt_char arg_path[PATH_CHARACTER_MAX] = {0};

int main(int argc, char* argv[]) {	
	if(argv[1]) {
		strcpy(arg_path, argv[1]);
	}else {
		printf(ERROR_CMDARGS);
		exit(1);
	}
	
	nn_start();
	//training_run();
	weights_save();
	nn_finish();

	return 0;
}
