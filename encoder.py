import argparse
import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('file_q')
arg_parser.add_argument('file_a')
arg_parser.add_argument('file_qa')
arg_parser.add_argument('file_vocab')
arg_parser.add_argument('file_weights')
args = arg_parser.parse_args()

with open(args.file_q, 'r') as file_q, open(args.file_a, 'r') as file_a, open(args.file_qa, 'w') as file_qa, open(args.file_vocab, 'r') as file_vocab, open(args.file_weights, 'r') as file_weights:
    vocab = {}
    vec_len = -1

    for line_vocab, line_weights in zip(file_vocab, file_weights):
        word, _ = line_vocab.strip().split('\t')
        vocab[word] = np.fromstring(line_weights.strip(), dtype=float, sep='\t')
        if vec_len == -1:
            vec_len = vocab[word].shape[0]
        elif vec_len != vocab[word].shape[0]:
            print("Vector length mismatch")
            exit(1)

    for q, a in zip(file_q, file_a):
        vec = np.zeros(vec_len)
        for word in q.split():
            if word in vocab:
                vec += vocab[word]
        file_qa.write(f'{q.strip()} {list(vec)} =~=~> {a.strip()}\n')
