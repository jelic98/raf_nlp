#define H_COL_IMPLEMENT
#define H_LOG_IMPLEMENT
#define H_STMR_IMPLEMENT
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

		printf("1:%s[END]\n", argv[1]);
		printf("2:%s[END]\n", argv[2]);
		printf("3:%s[END]\n", argv[3]);
	}else {
#ifdef FLAG_LOG
		echo_fail(ERROR_CMDARGS);
#endif
		exit(1);
	}
	
	nn_start();
	training_run();
	weights_save();
	nn_finish();

	return 0;
}
