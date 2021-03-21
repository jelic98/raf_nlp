#ifndef H_COL_INCLUDE
#define H_COL_INCLUDE

#include "lib.h"

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
xWord** filter;
dt_int filter_max;
#endif

void map_init(xWord**, dt_int**, dt_int);
xWord** map_get(xWord**, dt_int**, const dt_char*);
xWord* list_insert(xWord*, xWord**);
xContext* context_insert(xContext*, xWord*, dt_int*);
void context_flatten(xContext*, xWord**, dt_int*, dt_int*);

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
dt_int filter_contains(xWord**, const dt_char*);
#endif

void list_release(xWord*);
dt_int list_contains(xWord*, const dt_char*);
void context_release(xContext*);
xWord* bst_insert(xWord*, xWord**, dt_int*);
void bst_flatten(xWord*, xWord**, dt_int*);
xWord* node_create(const dt_char*);
xContext* node_context_create(xWord*);
void node_context_release(xContext*);
void node_release(xWord*);

#ifdef H_COL_IMPLEMENT
static dt_uint hash_get(const dt_char* word) {
	dt_ull i, h;

	for(h = i = 0; word[i]; i++) {
		h = h * 257 + word[i];
	}

	return h % VOCABULARY_HASH_MAX;
}

void map_init(xWord** vocab, dt_int** vocab_hash, dt_int size) {
	dt_uint p, h;

	for(h = 0; h < VOCABULARY_HASH_MAX; h++) {
		(*vocab_hash)[h] = -1;
	}

	for(p = 0; p < size; p++) {
		h = hash_get(vocab[p]->word);

		while((*vocab_hash)[h] != -1) {
			h = (h + 1) % VOCABULARY_HASH_MAX;
		}

		(*vocab_hash)[h] = p;
	}
}

xWord** map_get(xWord** vocab, dt_int** vocab_hash, const dt_char* word) {
	dt_uint h = hash_get(word);

	while(1) {
		if((*vocab_hash)[h] == -1) {
			return NULL;
		}

		if(!strcmp(word, vocab[(*vocab_hash)[h]]->word)) {
			return &vocab[(*vocab_hash)[h]];
		}

		h = (h + 1) % VOCABULARY_HASH_MAX;
	}

	return NULL;
}

xWord* list_insert(xWord* root, xWord** node) {
	if(root) {
		xWord* tmp = root;

		while(tmp->next) {
			if(!strcmp(tmp->word, (*node)->word)) {
				node_release(*node);
				*node = tmp;
				return root;
			}

			tmp = tmp->next;
		}

		tmp->next = *node;

		return root;
	} else {
		return *node;
	}

	return *node;
}

xContext* context_insert(xContext* root, xWord* word, dt_int* success) {
	*success = 0;

	if(root) {
		dt_int cmp = strcmp(root->word->word, word->word);

		if(cmp > 0) {
			root->left = context_insert(root->left, word, success);
		} else if(cmp < 0) {
			root->right = context_insert(root->right, word, success);
		} else {
			root->freq++;
		}

		return root;
	}

	*success = 1;
	return node_context_create(word);
}

void context_flatten(xContext* root, xWord** arr, dt_int* arr_freq, dt_int* index) {
	if(root) {
		context_flatten(root->left, arr, arr_freq, index);

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
		if(root->word->freq > 0) {
#endif
			arr[(*index)++] = root->word;
			arr_freq[(*index) - 1] = root->freq;
#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
		}
#endif

		context_flatten(root->right, arr, arr_freq, index);
	}
}

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
dt_int filter_contains(xWord** filter, const dt_char* word) {
	dt_int p;

	for(p = 0; p < filter_max; p++) {
		if(!strcmp(filter[p]->word, word)) {
			return 1;
		}
	}

	return 0;
}
#endif

#ifdef FLAG_FILTER_VOCABULARY_STOP
dt_int list_contains(xWord* root, const dt_char* word) {
	while(root) {
		if(!strcmp(root->word, word)) {
			return 1;
		}

		root = root->next;
	}

	return 0;
}
#endif

#ifdef FLAG_FREE_MEMORY
void list_release(xWord* root) {
	xWord* node;

	while(root) {
		node = root;
		root = root->next;
		node_release(node);
	}
}
#endif

void context_release(xContext* root) {
	if(root) {
		context_release(root->left);
		context_release(root->right);
		node_context_release(root);
	}
}

xWord* bst_insert(xWord* root, xWord** node, dt_int* success) {
	*success = 0;

	if(root) {
		dt_int cmp = strcmp(root->word, (*node)->word);

		if(cmp > 0) {
			root->left = bst_insert(root->left, node, success);
		} else if(cmp < 0) {
			root->right = bst_insert(root->right, node, success);
		} else {
			root->freq++;
			node_release(*node);
			*node = root;
		}

		return root;
	}

	(*node)->freq++;
	*success = 1;
	return *node;
}

void bst_flatten(xWord* root, xWord** arr, dt_int* index) {
	if(root) {
		bst_flatten(root->left, arr, index);

#if defined(FLAG_FILTER_VOCABULARY_LOW) || defined(FLAG_FILTER_VOCABULARY_HIGH)
		if(root->freq > 0) {
			arr[root->index = (*index)++] = root;
		}
#else
		arr[root->index = (*index)++] = root;
#endif

		bst_flatten(root->right, arr, index);
	}
}

xWord* node_create(const dt_char* word) {
	xWord* node = (xWord*) calloc(1, sizeof(xWord));
	memcheck(node);
	node->word = (dt_char*) calloc(strlen(word) + 2, sizeof(dt_char));
	memcheck(node->word);
	strcpy(node->word, word);
	node->word[strlen(node->word) + 1] = '*';
	node->index = node->prob = node->context_max = node->freq = 0;
	node->left = node->right = node->next = NULL;
	node->context = NULL;
	node->target = NULL;
	node->target_freq = NULL;
	return node;
}

xContext* node_context_create(xWord* word) {
	xContext* node = (xContext*) calloc(1, sizeof(xContext));
	memcheck(node);
	node->word = word;
	node->freq = 1;
	return node;
}

void node_release(xWord* root) {
	if(root->word && root->word[strlen(root->word) + 1] == '*') {
		free(root->word);
		root->word = NULL;
	}

	if(root->target) {
		free(root->target);
		root->target = NULL;
	}

	if(root->target_freq) {
		free(root->target_freq);
		root->target_freq = NULL;
	}

	root->left = root->right = root->next = NULL;
	root->index = root->prob = root->context_max = root->freq = 0;
	root->context = NULL;
	free(root);
}

void node_context_release(xContext* root) {
	root->word = NULL;
	root->left = root->right = NULL;
	free(root);
}
#endif
#endif
