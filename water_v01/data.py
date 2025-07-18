import os
import json
import pickle
import re
from collections import Counter

class Tokenizer:
    def __init__(self, min_freq=5, max_vocab=12000):
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.vocab = {'<pad>': 0, '<unk>': 1}
        self.id2token = {0: '<pad>', 1: '<unk>'}
        self.freqs = Counter()
        self.special_tokens = {'<pad>', '<unk>'}
        self.always_include = set(list('0123456789+-*/=.,:;?!()[]'))

    def normalize(self, text):
        # Lowercase
        text = text.lower()
        # Keep only allowed chars (letters, digits, whitespace, math symbols, common punct)
        text = re.sub(rf"[^a-z0-9\.\,\:\;\?\!\(\)\[\]\+\-\*\/\=\%\^\s]", "", text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        text = self.normalize(text)
        # Split on numbers, words, and single math/punct symbols
        tokens = re.findall(r"[0-9]+|[a-zA-Z]+|[+\-*/=.,:;?!()\[\]]", text)
        return tokens

    def build_vocab(self, dataset_dir, files, input_key='input', output_key='output'):
        print(f"[DEBUG] Building vocab from: {files}")
        for fname in files:
            path = os.path.join(dataset_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        for key in [input_key, output_key]:
                            text = self.normalize(obj.get(key, ''))
                            tokens = text.split()
                            self.freqs.update(tokens)
                    except Exception as e:
                        continue
        # Add tokens above min_freq, up to max_vocab, and always include math/digits/punct
        sorted_tokens = [tok for tok, freq in self.freqs.most_common() if (freq >= self.min_freq or tok in self.always_include) and tok not in self.special_tokens and tok != '<eos>']
        # Remove duplicates while preserving order
        seen = set()
        sorted_tokens = [x for x in sorted_tokens if not (x in seen or seen.add(x))]
        if len(sorted_tokens) > self.max_vocab - len(self.special_tokens) - 1:  # -1 for <eos>
            sorted_tokens = sorted_tokens[:self.max_vocab - len(self.special_tokens) - 1]
        for tok in sorted_tokens:
            idx = len(self.vocab)
            self.vocab[tok] = idx
            self.id2token[idx] = tok
        # Add <eos> as the last token
        eos_idx = len(self.vocab)
        self.vocab['<eos>'] = eos_idx
        self.id2token[eos_idx] = '<eos>'
        print(f"[DEBUG] Final vocab size: {len(self.vocab)} (including <eos>)")
        if len(self.vocab) > 50000:
            print(f"[WARNING] Vocab size is very large: {len(self.vocab)}")
        print("[DEBUG] Example tokens:", sorted_tokens[:20] + ['<eos>'])
        rare = [tok for tok, freq in self.freqs.items() if freq < self.min_freq and tok not in self.always_include]
        print(f"[DEBUG] Rare tokens (freq < {self.min_freq}): {rare[:10]}")

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(tok, self.vocab['<unk>']) for tok in tokens]

    def decode(self, ids):
        return ' '.join([self.id2token.get(i, '<unk>') for i in ids if i != self.vocab['<pad>']])

    def save_vocab(self, vocab_path='water_v01/tokenizer/vocab.pkl', id2token_path='water_v01/tokenizer/id2token.pkl'):
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.vocab, f)
        with open(id2token_path, 'wb') as f:
            pickle.dump(self.id2token, f)
        print(f"[DEBUG] Saved vocab to {vocab_path} and id2token to {id2token_path}")

    def load_vocab(self, vocab_path='water_v01/tokenizer/vocab.pkl', id2token_path='water_v01/tokenizer/id2token.pkl'):
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(id2token_path, 'rb') as f:
            self.id2token = pickle.load(f)
        print(f"[DEBUG] Loaded vocab from {vocab_path} and id2token from {id2token_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build and save tokenizer vocab for Water v0.1')
    parser.add_argument('--dataset_dir', type=str, default='E:/AI Work Space/code playground/dataset maker for Braham water10m/water_demonstrator_dataset/', help='Path to dataset directory')
    parser.add_argument('--input_key', type=str, default='prompt', help='Key for input field in JSONL')
    parser.add_argument('--output_key', type=str, default='response', help='Key for output field in JSONL')
    parser.add_argument('--min_freq', type=int, default=5, help='Minimum frequency for tokens')
    parser.add_argument('--max_vocab', type=int, default=12000, help='Maximum vocab size')
    args = parser.parse_args()
    files = [f for f in os.listdir(args.dataset_dir) if f.endswith('.jsonl')]
    # Print first 2 lines of each file for debug
    for fname in files:
        path = os.path.join(args.dataset_dir, fname)
        print(f'\n[DEBUG] {fname} first 2 lines:')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for i in range(2):
                    line = next(f)
                    obj = json.loads(line)
                    print(f'  prompt: {obj.get(args.input_key, "")}')
                    print(f'  response: {obj.get(args.output_key, "")}')
        except Exception as e:
            print(f'  [ERROR] {e}')
    tokenizer = Tokenizer(min_freq=args.min_freq, max_vocab=args.max_vocab)
    tokenizer.build_vocab(args.dataset_dir, files, input_key=args.input_key, output_key=args.output_key)
    tokenizer.save_vocab() 