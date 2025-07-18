import torch
from model import BrahmaModel
from data import Tokenizer
import config
import sys
import os
import pickle
import torch.nn.functional as F

def sample_logits(logits, top_k=30, temperature=0.8, greedy=False):
    if greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    values, _ = torch.topk(logits, top_k)
    min_threshold = values[:, -1].unsqueeze(1)
    logits[logits < min_threshold] = -float('inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def load_tokenizer():
    """Load tokenizer from vocab.pkl and id2token.pkl. Raise error if missing."""
    tokenizer = Tokenizer()
    vocab_path = 'water_v01/tokenizer/vocab.pkl'
    id2token_path = 'water_v01/tokenizer/id2token.pkl'
    if not os.path.exists(vocab_path) or not os.path.exists(id2token_path):
        raise FileNotFoundError(f"Tokenizer vocab/id2token not found at {vocab_path} and {id2token_path}. Please run data.py to generate vocab.")
    tokenizer.load_vocab(vocab_path, id2token_path)
    return tokenizer

def load_model(tokenizer):
    model = BrahmaModel(len(tokenizer.vocab), d_model=config.D_MODEL, n_layers=config.N_LAYERS, n_heads=config.N_HEADS, d_ff=config.D_FF, max_seq_len=config.MAX_SEQ_LEN, dropout=config.DROPOUT)
    model.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.to(config.DEVICE)
    model.eval()
    return model

def generate(prompt, max_new_tokens=100, top_k=30, temperature=0.8, repetition_penalty=1.2, greedy=False):
    tokenizer = load_tokenizer()
    model = load_model(tokenizer)
    input_ids = tokenizer.encode(prompt)
    print(f"[DEBUG] Tokenized input: {input_ids}")
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=config.DEVICE)
    generated = input_ids.copy()
    eos_token_id = tokenizer.vocab.get('<eos>', None)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_tensor)
            next_token_logits = logits[:, -1, :]
            # Repetition suppression: penalize tokens in last 10
            for token_id in set(generated[-10:]):
                next_token_logits[:, token_id] *= 0.5
            next_token = sample_logits(next_token_logits, top_k=top_k, temperature=temperature, greedy=greedy)
            next_token_id = next_token.item()
            generated.append(next_token_id)
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            input_tensor = torch.tensor([generated], dtype=torch.long, device=config.DEVICE)
            if input_tensor.size(1) >= config.MAX_SEQ_LEN:
                break
    print(f"[DEBUG] Model output token IDs: {generated}")
    output = tokenizer.decode(generated)
    print(f"[DEBUG] Decoded output: {output}")
    return output

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Water v0.1 Inference')
    parser.add_argument('prompt', nargs='*', help='Prompt for the model')
    parser.add_argument('--greedy', action='store_true', help='Use greedy decoding (argmax)')
    parser.add_argument('--math', action='store_true', help='Math prompt mode (prints prompt and output)')
    args = parser.parse_args()
    prompt = ' '.join(args.prompt) if args.prompt else input('Prompt: ')
    output = generate(prompt, greedy=args.greedy)
    if args.math:
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
    else:
        print(output) 