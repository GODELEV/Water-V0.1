import torch

# Paths
DATASET_DIR = r"E:/AI Work Space/code playground/dataset maker for Braham water10m/water_demonstrator_dataset/"
DATASET_FILES = [
    "math_reasoning.jsonl",
    "logic_qa.jsonl",
    "instructions.jsonl",
    "dialogue.jsonl",
    "code.jsonl",
    "multilingual.jsonl"
]
BEST_MODEL_PATH = "water_v01/water_best.pt"
LOG_PATH = "water_v01/train_log.txt"

# Model hyperparameters
# 5M param config
VOCAB_SIZE = 12000  # Set by tokenizer build, max_vocab=12000
D_MODEL = 192
N_LAYERS = 4
N_HEADS = 3
D_FF = 768
MAX_SEQ_LEN = 128
DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 32  # Will be dynamically adjusted
EPOCHS = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Monitoring
MONITOR_PORT = 6006 