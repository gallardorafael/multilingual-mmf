import numpy as np
import argparse

from nltk.tokenize import word_tokenize
from collections import Counter

parser = argparse.ArgumentParser(description='Obtain a list of tokens given a frequency.')
parser.add_argument('--input_path', help='Path to the annotation file (npy).')
parser.add_argument('--output_path', help='Path to save the vocabulary (txt).')
parser.add_argument('--min_freq', help='Min frequency of tokens to be added to vocab.')
args = parser.parse_args()

def build_vocab(annotation_path):
    annotation = np.load(annotation_path, allow_pickle=True)
    print("File",annotation_path,"succesfully loaded.")
    counter = Counter()
    for caption in annotation[1:]:
        caption_tokens = word_tokenize(caption['caption_str'])
        counter.update([word.lower() for word in caption_tokens])

    return counter

def vocab_to_txt(counter, output_path, min_freq):
    # Special tokens
    tokens=['<pad>', '<s>', '</s>', '<unk>']
    # Sorting by frequency
    vocab = sorted(counter, key=counter.get, reverse=True)
    # Obtaining tokens that appears more than min_freq in dataset
    vocab = tokens+[word for word in vocab if counter[word] >= min_freq]

    # Writing to txt file
    with open(output_path, 'w') as f:
        f.writelines("%s\n" % word for word in vocab)

    print(output_path,"wrote succesfully.")
def main():
    counter = build_vocab(args.input_path)
    vocab_to_txt(counter, args.output_path, int(args.min_freq))

if __name__ == "__main__":
    main()
