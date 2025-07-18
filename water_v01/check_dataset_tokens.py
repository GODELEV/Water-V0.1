import os
import json
from collections import Counter

data_dir = r"E:/AI Work Space/code playground/dataset maker for Braham water10m/water_demonstrator_dataset/"
files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]

for fname in files:
    print(f"\n=== {fname} ===")
    path = os.path.join(data_dir, fname)
    with open(path, 'r', encoding='utf-8') as f:
        lines = [next(f) for _ in range(3)]
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
            prompt = obj.get('prompt', '')
            response = obj.get('response', '')
            print(f"Sample {i+1} prompt: {prompt}")
            print(f"Sample {i+1} response: {response}")
        except Exception as e:
            print(f"Error reading line {i+1}: {e}")
    # Token check
    tokens = set()
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i > 100: break
            try:
                obj = json.loads(line)
                tokens.update(obj.get('prompt', '').strip().split())
                tokens.update(obj.get('response', '').strip().split())
            except: pass
    print(f"Unique tokens in first 100 lines: {list(tokens)[:20]} (total: {len(tokens)})") 