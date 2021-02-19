import np
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('file_q')
arg_parser.add_argument('file_a')
arg_parser.add_argument('file_qa')
arg_parser.add_argument('file_vocab')
arg_parser.add_argument('file_weights')
args = arg_parser.parse_args()

with(open(args.file_q, 'r') as file_q,
        open(args.file_a, 'r') as file_a,
        open(args.file_aq, 'w') as file_qa,
        open(args.file_vocab, 'rb') as file_vocab,
        open(args.file_weights, 'rb') as file_weights):
    vocab = {}

    for line_vocab, line_weights in zip(file_vocab, file_weights):
        word, _ = line_vocab.split('\t')
        vocab[word] = np.fromstring(line_weights, dtype=float, sep='\t')

    for q, a in zip(file_q, file_a):
        vec = reduce(lambda x,y: x+vocab[y], q.split())
        file_qa.write(f'{q} ({vec}) =~=~> {a}\n')
