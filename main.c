#include "include/main.h"
#include "include/nn.h"

int main() {
	//load_weights();
	start_training();
	save_weights();
	get_predictions("agama", 5, 10);
	finish_training();

	return 0;
}
