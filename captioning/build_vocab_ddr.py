import nltk
import pickle
import argparse
from collections import Counter
import pandas as pd
import json


class Vocabulary(object):
    
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(csv, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    
    #We have pruned the data to use fewer songs (scores of 5-9)
    all_data = pd.read_csv(csv, index_col=False)
    all_data['json'] = all_data.apply(lambda x: x.to_json(), axis=1)
    all_data = [json.loads(row) for row in all_data['json']]

    for i, row in enumerate(all_data):

        #Change semicolon to space
        row['text'] = row['text'].split(';')
        row['text'] = ' '.join(row['text'])
        caption = str(row['text'])
        #print(caption)
        tokens = nltk.tokenize.word_tokenize(caption)
        #print(tokens)
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    
    for word, cnt in counter.items():
        print(word, cnt)

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(csv=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/spectrogram_2.csv', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab_ddr.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=7000, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
    