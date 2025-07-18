import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from model import BrahmaModel
from data import Tokenizer
from monitor import LiveMonitor
import config
import os
from tqdm import tqdm
import pickle
import json
import re
import math

def parse_train_plan(plan_path):
    plan = []
    with open(plan_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Format: dataset | epochs | lr | batch_size
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 2:
                continue
            dataset = parts[0]
            epochs = int(parts[1])
            lr = float(parts[2]) if len(parts) > 2 and re.match(r'^[0-9.eE-]+$', parts[2]) else config.LEARNING_RATE
            batch_size = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else config.BATCH_SIZE
            plan.append({'dataset': dataset, 'epochs': epochs, 'lr': lr, 'batch_size': batch_size})
    return plan

class JSONLDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        response = item.get('response', '')
        if not response.endswith(' <eos>'):
            response = response.strip() + ' <eos>'
        text = item.get('prompt', '') + ' ' + response
        ids = self.tokenizer.encode(text)
        if len(ids) < self.max_len:
            ids += [self.tokenizer.vocab['<pad>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

def get_dataloader(file_path, tokenizer, batch_size, max_len=128, shuffle=True):
    dataset = JSONLDataset(file_path, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train():
    # Load tokenizer vocab
    vocab_path = 'water_v01/tokenizer/vocab.pkl'
    id2token_path = 'water_v01/tokenizer/id2token.pkl'
    tokenizer = Tokenizer()
    tokenizer.load_vocab(vocab_path, id2token_path)
    vocab_size = len(tokenizer.vocab)
    print("[DEBUG] Final vocab size:", vocab_size)
    print("[DEBUG] Sample vocab:", list(tokenizer.vocab.items())[:20])

    # Model
    model = BrahmaModel(vocab_size, d_model=config.D_MODEL, n_layers=config.N_LAYERS, n_heads=config.N_HEADS, d_ff=config.D_FF, max_seq_len=config.MAX_SEQ_LEN, dropout=config.DROPOUT)
    model.to(config.DEVICE)

    # Curriculum plan
    plan = parse_train_plan('water_v01/train_plan.txt')
    monitor = LiveMonitor()
    best_val_loss = float('inf')

    for phase in plan:
        fname = phase['dataset']
        epochs = phase['epochs']
        lr = phase['lr']
        batch_size = phase['batch_size']
        print(f"\n=== Training on {fname} | epochs={epochs} | lr={lr} | batch_size={batch_size} ===")
        dataloader = get_dataloader(os.path.join(config.DATASET_DIR, fname), tokenizer, batch_size, max_len=config.MAX_SEQ_LEN)
        n = len(dataloader.dataset)
        n_val = max(1, int(0.05 * n))
        n_train = n - n_val
        train_set, val_set = random_split(dataloader.dataset, [n_train, n_val])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        # Update optimizer for new LR
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
        scaler = GradScaler()
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<pad>'])
        # Print sample input/target
        sample_batch = next(iter(train_loader))
        input_ids = sample_batch[:, :-1]
        target_ids = sample_batch[:, 1:]
        print("[DEBUG] Sample input:", input_ids[0])
        print("[DEBUG] Sample target:", target_ids[0])
        for epoch in range(1, epochs+1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                batch = batch.to(config.DEVICE)
                optimizer.zero_grad()
                with autocast():
                    logits = model(batch[:, :-1])
                    loss = criterion(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                # Accuracy calculation
                preds = logits.argmax(dim=-1)
                targets = batch[:, 1:]
                mask = (targets != tokenizer.vocab['<pad>'])
                correct += ((preds == targets) & mask).sum().item()
                total += mask.sum().item()
                pbar.set_postfix({'loss': loss.item()})
                if hasattr(monitor, '_draw'):
                    monitor._draw()
            avg_train_loss = running_loss / len(train_loader)
            train_acc = correct / total if total > 0 else 0.0

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(config.DEVICE)
                    with autocast():
                        logits = model(batch[:, :-1])
                        loss = criterion(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
                    val_loss += loss.item()
                    preds = logits.argmax(dim=-1)
                    targets = batch[:, 1:]
                    mask = (targets != tokenizer.vocab['<pad>'])
                    val_correct += ((preds == targets) & mask).sum().item()
                    val_total += mask.sum().item()
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total if val_total > 0 else 0.0

            # Monitor
            monitor.update(avg_train_loss, avg_val_loss, train_acc, val_acc, epoch, fname)
            perplexity = math.exp(avg_val_loss)
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Perplexity: {perplexity:.4f}")
            with open(config.LOG_PATH, 'a') as f:
                f.write(f"{fname}, epoch {epoch}, train {avg_train_loss:.4f}, val {avg_val_loss:.4f}, train_acc {train_acc:.4f}, val_acc {val_acc:.4f}, ppl {perplexity:.4f}\n")

            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save(config.BEST_MODEL_PATH)
                print(f"[Best model saved @ {config.BEST_MODEL_PATH}]")

    monitor.close()
    print("Training complete.")

if __name__ == '__main__':
    train() 