import gdown
import argparse
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('url')
arg_parser.add_argument('file_d_out')
arg_parser.add_argument('file_s_out')
args = arg_parser.parse_args()

gdown.download(args.url, args.file_d_out, quiet=False)

with open(args.file_s_out, 'w') as file_s_out:
    for word in set(stopwords.words('english')):
        file_s_out.write(word + '\n')
