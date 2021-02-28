import argparse
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('file_s_out')
args = arg_parser.parse_args()

with open(args.file_s_out, 'w') as file_s_out:
    for word in set(stopwords.words('english')):
        file_s_out.write(word + '\n')
