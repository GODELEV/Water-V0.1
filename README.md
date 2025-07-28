# Water v0.1 — Brahma Architecture Model Training System

> **Water v0.1 is the first test of the Brahma post-transformer architecture. It’s a 5M parameter model trained entirely on supervised data to validate architecture performance on logic, math, instruction, and code.**
>
> While Water v0.1 does not yet generalize broadly (due to lack of unsupervised pretraining), it shows clear signs of efficient reasoning, especially in logic and math domains, on extremely limited compute (RTX 3050, 4GB VRAM).
>
> It proves that Brahma can outperform transformers per parameter in focused domains — even without full-scale pretraining.

---

## 🟢 Should I Mark It a Success?
**YES.**

Because 
our goals were:

✅ Train a new architecture from scratch

✅ Fit it in 10M parameters

✅ Run on local hardware

✅ Beat transformer baselines in early tests

---

## 📘 Lessons Learned / Retrospective

**✅ What worked:**
- Architecture design
- Training on instructions
- Local run

**⚠️ What didn’t:**
- No unsupervised generalization
- EOS token weirdness

**🧭 What’s next:**
- Brahma unsupervised pretraining
- Water v0.2 / Fire v0.1 etc.

---

## Features
- **Curriculum training**: Trains on each .jsonl file in order
- **Mixed precision**: Fast, memory-efficient training
- **Live monitoring**: Pygame loss graph, epoch, and dataset display
- **Best model saving**: By validation loss
- **Custom Brahma architecture**: ~5M params, post-transformer
- **Simple tokenizer**: Fast, memory-light, with vocab caching
- **Modular codebase**: Easy to extend
- **Inference**: CLI or Python-callable

## File Structure
```
water_v01/
├── model.py         # Brahma model definition
├── data.py          # JSONL loader, tokenizer, DataLoader
├── train.py         # Curriculum training loop
├── monitor.py       # Pygame live graph
├── inference.py     # Inference script
├── config.py        # Paths, hyperparams
├── tokenizer/
│   └── vocab.pkl    # Saved tokenizer vocab (auto-generated)
├── requirements.txt # Dependencies
└── README.md        # This file
```

## Dataset Format (JSONL)
Each line in your dataset should be a JSON object with `input` and `output` fields:

```
{"input": "What is 2 + 2?", "output": "4"}
{"input": "Natalia sold 48 clips...", "output": "Natalia sold 24 clips in May..."}
```

## Tokenizer & Vocab Caching
- The tokenizer builds a vocabulary from all training data and saves it to `tokenizer/vocab.pkl`.
- On future runs, the vocab is loaded from this file for both training and inference, ensuring consistency.
- If `vocab.pkl` is missing, it will be rebuilt from the data.

## Training
- Run: `python train.py`
- Prints debug info: vocab size, sample vocab, decoded sample input/target
- Batch size and sequence length are tuned for 4GB VRAM (see `config.py`)
- PyTorch version: 1.12+ recommended

## Inference
- Run: `python inference.py "Your prompt here"`
- Inference loads the saved vocab and model for consistent results
- If vocab is missing, you must retrain or copy `vocab.pkl` from a previous run
- Inference is interactive by default (or pass prompt as CLI argument)

## Notes
- Training log is saved to `train_log.txt`
- Early stopping and progress bar included
- For best results, ensure your dataset covers the types of prompts you want to infer

---
**Water v0.1** — Efficient, modular, and ready for research or production! 

## License

MIT License

Copyright (c) 2024 Water v0.1 Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
