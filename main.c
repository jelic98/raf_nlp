#define H_LOG_IMPLEMENT
#define H_MAT_IMPLEMENT
#define H_COL_IMPLEMENT
#define H_VOC_IMPLEMENT
#define H_SENT_IMPLEMENT
#define H_STMR_IMPLEMENT
#include "nn.h"

dt_char arg_actions[PATH_CHARACTER_MAX] = {0};
dt_char arg_train[PATH_CHARACTER_MAX] = {0};
dt_char arg_test[PATH_CHARACTER_MAX] = {0};
dt_char arg_stop[PATH_CHARACTER_MAX] = {0};

int main(int argc, char* argv[]) {
	if(argc == ARGS_COUNT) {
		strcpy(arg_actions, argv[1]);
		strcpy(arg_train, argv[2]);
		strcpy(arg_test, argv[3]);
		strcpy(arg_stop, argv[4]);
	}else {
#ifdef FLAG_LOG
		echo_fail(ERROR_CMDARGS);
#endif
		exit(1);
	}

	nn_start();
	
	dt_char* action = strtok(arg_actions, ",");

	while(action) {
		if(!strcmp(action, "TRAIN")) {
			training_run();
		}else if(!strcmp(action, "TEST")) {
			testing_run();
		}else if(!strcmp(action, "LOAD")) {
			weights_load();
		}else if(!strcmp(action, "SAVE")) {
			weights_save();
		}

		action = strtok(NULL, ",");
	}

	nn_finish();

	return 0;
}
