#include "include/main.h"
#include "include/data.h"
#include "include/nn.h"

int main() {
	text_to_sentences();
	sentences_to_words();
	words_to_onehot();
//	start_training();
	
	return 0;
}
