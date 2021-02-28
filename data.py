import gdown
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('url')
arg_parser.add_argument('file_d_out')
args = arg_parser.parse_args()

gdown.download(args.url, args.file_d_out, quiet=False)
