#include "include/data.h"

typedef struct xWord {
	char* word;
	int count;
	struct xWord* left;
	struct xWord* right;
} xWord;

typedef struct xBit {
	unsigned int on : 1;
} xBit;

static char words[WORDS_MAX][WORD_MAX];
static int dict_size;

static xWord* bst_insert(xWord* node, char *str) {
	if(!node) {
		node = (xWord*) malloc(sizeof(xWord));
		node->word = str;
		node->left = node->right = NULL;
		node->count = 1;
		
		return node;
	}

	int cmp = strcmp(str, node->word);
	
	if(cmp < 0) {
		node->left = bst_insert(node->left, str);
	}else if(cmp > 0) {
		node->right = bst_insert(node->right, str);
	}else {
		node->count++;
	}

	return node;
}

static xWord* word_arr[WORDS_MAX];

static void bst_to_matrix(xWord* node) {
	if(node) {
		bst_to_matrix(node->left);

		word_arr[dict_size++] = node;

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

		node->word = NULL;
		node->count = 0;
		node = NULL;
	}
}

void text_to_sentences() {
	FILE* fin = fopen(CORPUS_IN_FILE, "r");
	FILE* fout = fopen(CORPUS_OUT_FILE, "w");

	if(!fin || !fout) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
		return;
	}

	int sep = 0, dot = 0;
	char c;

	while((c = fgetc(fin)) != EOF) {
		if(c == ' ' || c == '\t') {
			sep = 1;
			continue;
		}
		
		if(c == '.') {
			dot = 1;
			continue;
		}
		
		if(dot) {
			fprintf(fout, "\n");
		}else if(sep) {
			fprintf(fout, " ");
		}	
		
		sep = dot = 0;

		if(isalnum(c)) {
			fprintf(fout, "%c", tolower(c));
		}
	}

	if(fclose(fin) == EOF || fclose(fout) == EOF) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
	}
}

void sentences_to_words() {
	FILE* fin = fopen(CORPUS_OUT_FILE, "r");

	if(!fin) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
		return;
	}

	int i = 0, sep = 0;
	char c;
	char *pc = words[i];

	while((c = fgetc(fin))) {
		if(c == ' ' || c == '\n') {
			sep = 1;
			continue;
		}

		if(sep) {
			pc = words[++i];
		}
			
		if(c == EOF) {
			break;
		}
		
		sep = 0;
		
		*pc++ = c;
	}

	if(fclose(fin) == EOF) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
	}
}

void words_to_onehot() {
	xWord* root = NULL;
	
	int i = 0;

	while(*words[i]) {
		root = bst_insert(root, words[i++]);
	}

	bst_to_matrix(root);

	for(i = 0; i < dict_size; i++) {
		printf("%d %s\n", i, word_arr[i]->word);
	}

	bst_clear(root);

	return;
	
	xBit onehot[dict_size][dict_size];
	
	memset(onehot, 0, sizeof(onehot));
	
	for(i = 0; i < dict_size; i++) {
		onehot[i][i].on = 1;
	}

	int j;

	for(i = 0; i < dict_size; i++) {	
		printf("%d.\t", i + 1);
		
		for(j = 0; j < dict_size; j++) {
			printf("%d", onehot[i][j].on);
		}
		
		printf(" : %s\n", words[i]);
	}
}
