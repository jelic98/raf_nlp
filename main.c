// https://towardsdatascience.com/skip-gram-nlp-context-words-prediction-algorithm-5bbf34f84e0c

#include "include/main.h"
#include "include/nn.h"

int main() {
	//load_weights();
	start_training();
	save_weights();
	get_predictions("president", 10);
	finish_training();

	return 0;
}
