import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('file_q_in')
arg_parser.add_argument('file_a_in')
arg_parser.add_argument('file_q_out')
arg_parser.add_argument('file_a_out')
arg_parser.add_argument('freq_min')
arg_parser.add_argument('freq_top_cut')
args = arg_parser.parse_args()

stop_words = set(stopwords.words('english'))

with open(args.file_q_in, 'r') as file_q_in, open(args.file_a_in, 'r') as file_a_in, open(args.file_q_out, 'w') as file_q_out, open(args.file_a_out, 'w') as file_a_out:
    questions, answers = [], []

    for q, a in zip(file_q_in, file_a_in):
        tokens = word_tokenize(q)
        questions.append([t for t in tokens if t not in stop_words])
        answers.append(a)

    vocab = {}

    for line in questions:
        for token in line:
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1

    freqs = sorted(vocab.items(), key=lambda x: x[1])
    freqs = list(filter(lambda x: x[1] > int(args.freq_min), freqs))
    freq_top_cut = int(int(args.freq_top_cut) * len(freqs) / 100)
    freqs = freqs[:-freq_top_cut]
    
    vocab = [k for k, v in freqs]

    for q, a in zip(questions, answers):
        q = list(filter(lambda x: x in vocab, q))
        if len(q) > 0:
            file_q_out.write(' '.join(q) + '\n')
            file_a_out.write(a)
