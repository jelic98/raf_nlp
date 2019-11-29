#include "include/main.h"
#include "include/nn.h"

int main() {
	start_training();
	get_predictions("animal", 10);
	finish_training();

	return 0;
}
