#include "include/data.h"

// TODO Make vector using array index just before putting it into network

static char sentences[SENTENCE_MAX][WORD_MAX][CHARACTER_MAX];
static int dict_size;
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

static void bst_to_matrix(xWord* node, xWord** words) {
	if(node) {
		bst_to_matrix(node->left, words);

		words[dict_size++] = node;

		bst_to_matrix(node->right, words);
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

static void file_cleanup() {
	FILE* fin = fopen(CORPUS_FILE, "r");
	FILE* fout = fopen(CORPUS_CLEAN_FILE, "w");

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

void prepare_data(xWord** words) {
	file_cleanup();

	FILE* fin = fopen(CORPUS_CLEAN_FILE, "r");

	if(!fin) {
		fprintf(LOG_FILE, FILE_ERROR_MESSAGE);
		return;
	}

	char line[WORD_MAX * CHARACTER_MAX];
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
			
			root = bst_insert(root, sentences[i][j]);
			
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

	bst_to_matrix(root, words);

	xBit onehot[dict_size][dict_size];
	memset(onehot, 0, sizeof(onehot));
	
	for(i = 0; i < dict_size; i++) {
		onehot[i][i].on = 1;
	}

	for(i = 0; i < dict_size; i++) {	
		printf("%d.\t", i + 1);
		
		for(j = 0; j < dict_size; j++) {
			printf("%d", onehot[i][j].on);
		}
		
		printf(" : %s (%d)\n", words[i]->word, words[i]->count);
	}

	bst_clear(root);
}
