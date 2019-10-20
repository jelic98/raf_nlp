#include "include/data.h"

// TODO Make vector using array index just before putting it into network

static int dict_size, sent_count;
static char context[SENTENCE_MAX][WORD_MAX][CHARACTER_MAX];
static xBit onehot[SENTENCE_MAX * WORD_MAX][SENTENCE_MAX * WORD_MAX];
static xWord* root = NULL;
static xWord* words[SENTENCE_MAX * WORD_MAX];

static xWord* bst_insert(xWord* node, const char* word) {
	if(!node) {
		node = (xWord*) malloc(sizeof(xWord));
		strcpy(node->word, word);
		node->left = node->right = NULL;
		node->count = 1;
		return node;
	}

	int cmp = strcmp(word, node->word);
	
	if(cmp < 0) {
		node->left = bst_insert(node->left, word);
	}else if(cmp > 0) {
		node->right = bst_insert(node->right, word);
	}else {
		node->count++;
	}

	return node;
}

static void bst_to_matrix(xWord* node) {
	if(node) {
		bst_to_matrix(node->left);
		words[dict_size++] = node;
		bst_to_matrix(node->right);
	}
}

static void bst_clear(xWord* node) {
	if(node) {
		bst_clear(node->left);
		bst_clear(node->right);
		node->left = NULL;
		node->right = NULL;
		free(node);
		memset(node->word, 0, sizeof(node->word));
		node->count = 0;
		node = NULL;
	}
}

static void read_file() {
	FILE* fin = fopen(CORPUS_FILE, "r");

	if(!fin) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
		return;
	}

	int i = 0, j = 0;
	char c, word[WORD_MAX] = {0};
	char* pw = word;

	while((c = fgetc(fin)) != EOF) {
		if(isalnum(c)) {
			*pw++ = tolower(c);
		}else if(!isalnum(c) && word[0]) {
			strcpy(context[i][j++], word);
			root = bst_insert(root, word);
			memset(pw = word, 0, sizeof(word));
		}else if(c == '.') {
			sent_count = ++i, j = 0;
		}	
	}

	if(fclose(fin) == EOF) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
	}
}

static void build_onehots() {
	int i;

	bst_to_matrix(root);

	for(i = 0; i < dict_size; i++) {
		onehot[i][i].on = 1;
	}
}

void prepare_data() {
	read_file();
	build_onehots();

	int i, j, k;
	
	for(i = 0, j = -1; i < sent_count; i++, j = -1) {
		while(j++, context[i][j][0]) {
			printf("%s:", context[i][j]);
			
			for(k = j - WINDOW_MAX; k <= j + WINDOW_MAX; k++) {
				if(k != j && k > 0 && context[i][k][0]) {
					printf(" %s", context[i][k]);
				}
			}
		
			printf("\n");
		}
	}
	
	for(i = 0; i < dict_size; i++) {	
		printf("%s%d.\t", i ? "" : "\n", i + 1);
		
		for(j = 0; j < dict_size; j++) {
			printf("%d", onehot[i][j].on);
		}
		
		printf(" : %s (%d)\n", words[i]->word, words[i]->count);
	}

	bst_clear(root);
}
