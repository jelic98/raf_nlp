#include "include/data.h"

// TODO Make vector using array index just before putting it into network

typedef struct xWord {
	char word[WORD_MAX];
	unsigned int count;
	struct xWord* left;
	struct xWord* right;
} xWord;

typedef struct xBit {
	unsigned int on : 1;
} xBit;

static char sentences[SENTENCES_MAX][SENTENCE_MAX][WORD_MAX];
static int dict_size;
static xWord* word_arr[WORDS_MAX];
static xWord* root = NULL;

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

		memset(node->word, 0, sizeof(node->word));
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
		if(isspace(c)) {
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

void sentences_to_matrix() {
	FILE* fin = fopen(CORPUS_OUT_FILE, "r");

	if(!fin) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
		return;
	}

	char line[SENTENCE_MAX];
	char* pl;
	char sep[] = " ";
	char dot[] = "\n";
	char* word;
	int i = 0, j, k;

	while(fgets(line, sizeof(line), fin)) {
		pl = strtok(line, dot);
		word = strtok(pl, sep);

		j = 0;

		while(word) {
			strcpy(sentences[i][j++], word);

			word = strtok(NULL, sep);
		}

		i++;
	}

	int total_i = i;

	for(i = 0; i < total_i; i++) {
		j = 0;

		while(sentences[i][j][0]) {
			printf("%s: ", sentences[i][j]);
			
			for(k = j - WINDOW_MAX; k <= j + WINDOW_MAX; k++) {
				if(k == j || k < 0 || !sentences[i][k][0]) {
					continue;
				}

				printf("%s ", sentences[i][k]);
			}
		
			printf("\n");

			j++;
		}
	}

	if(fclose(fin) == EOF) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
	}
}

void sentences_to_words() {
	FILE* fin = fopen(CORPUS_OUT_FILE, "r");

	if(!fin) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
		return;
	}

	char word[WORD_MAX];

	while(fscanf(fin, "%s", word) != EOF) {
		root = bst_insert(root, word);
	}

	if(fclose(fin) == EOF) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
	}
}

void words_to_onehot() {
	bst_to_matrix(root);

	xBit onehot[dict_size][dict_size];
	memset(onehot, 0, sizeof(onehot));
	
	int i, j;
	
	for(i = 0; i < dict_size; i++) {
		onehot[i][i].on = 1;
	}

	return;

	for(i = 0; i < dict_size; i++) {	
		printf("%d.\t", i + 1);
		
		for(j = 0; j < dict_size; j++) {
			printf("%d", onehot[i][j].on);
		}
		
		printf(" : %s (%d)\n", word_arr[i]->word, word_arr[i]->count);
	}

	bst_clear(root);
}
