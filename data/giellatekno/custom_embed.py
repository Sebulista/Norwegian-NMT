""" Embed sentences using a sentencetransformer model """

import tensorflow
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser, FileType
import sys

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("input", type=FileType('r'), help="File or stdin with new line separated sentences")
    parser.add_argument("output", type=FileType('wb', 0), help="Name of file to write to")

    return parser.parse_args()


def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer

def main():
    args = parse_arguments()

    model_folder = "path/to/models"

    #model = SentenceTransformer(f"{model_folder}/distiluse-base-multilingual-cased")
    #model = SentenceTransformer(f"{model_folder}/distiluse-base-multilingual-cased-v2")
    #model = SentenceTransformer(f"{model_folder}/paraphrase-multilingual-MiniLM-L12-v2")
    
    #model = SentenceTransformer(f"{model_folder}/paraphrase-multilingual-mpnet-base-v2")
    
    # LaBSE is recommended
    model = SentenceTransformer(f"{model_folder}/LaBSE")
    
    #sents = args.input()

    for sentences in buffered_read(args.input, 10000):
        encoded = model.encode(sentences)
        encoded.tofile(args.output)

if __name__ == "__main__":
    main()    
