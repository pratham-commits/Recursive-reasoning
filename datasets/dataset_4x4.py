import torch
from torch.utils.data import Dataset
import random
import numpy as np

VOCAB = ["<PAD>", "<SOS>", "<EOS>", 
         "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
         "a", "b", "c", "x", "y", "z", 
         "=", "+", "-", "*", ";", "print", "(", ")", 
         "if", "else", ":", ">", "<", "|", " "]

TOKEN_TO_IDX = {t: i for i,t in enumerate(VOCAB)}
IDX_TO_TOKEN = {i: t for i,t in enumerate(VOCAB)}

class GeneralTokenizer:
    def encode(self, text, max_len=64):
        for s in ["=", "+", "-", "*", ";", "(", ")", ":", ">", "<", "|"]:
            text = text.replace(s, f" {s} ")
        for d in "0123456789":
            text = text.replace(d, f" {d} ")
        tokens = text.split()
        ids = [TOKEN_TO_IDX.get(t, TOKEN_TO_IDX["<PAD>"]) for t in tokens]
        ids = [TOKEN_TO_IDX["<SOS>"]] + ids + [TOKEN_TO_IDX["<EOS>"]]
        if len(ids) < max_len:
            ids += [TOKEN_TO_IDX["<PAD>"]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids):
        tokens = []
        for i in ids:
            t = IDX_TO_TOKEN.get(int(i), "")
            if t == "<EOS>" or t == "<PAD>": break
            if t != "<SOS>": tokens.append(t)
        return "".join(tokens).replace(" ; ", "; ").replace(" = ", "=").replace(" | ", "|")

def generate_sudoku_base():
    """Generates a valid full 4x4 grid"""
    base = np.array([[1, 2, 3, 4],
                     [3, 4, 1, 2],
                     [2, 1, 4, 3],
                     [4, 3, 2, 1]])
    if random.random() > 0.5: base[[0, 1]] = base[[1, 0]]
    if random.random() > 0.5: base[[2, 3]] = base[[3, 2]]
    if random.random() > 0.5: base[:, [0, 1]] = base[:, [1, 0]]
    if random.random() > 0.5: base[:, [2, 3]] = base[:, [3, 2]]
    return base

def generate_sudoku_level(difficulty="medium"):
    base = generate_sudoku_base()
    problem = base.copy()
    
    # Logic for removing cells
    if difficulty == "easy":
        num_missing = random.randint(4, 5)
    elif difficulty == "hard":
        num_missing = random.randint(9, 11) # Very sparse
    else: # Medium
        num_missing = random.randint(6, 8)
        
    # Create random mask
    indices = [(r, c) for r in range(4) for c in range(4)]
    random.shuffle(indices)
    
    for i in range(num_missing):
        r, c = indices[i]
        problem[r, c] = 0
        
    input_str = "|".join(["".join(map(str, row)) for row in problem])
    target_str = "|".join(["".join(map(str, row)) for row in base])
    
    return input_str, target_str

# Maintain backward compatibility
def generate_sudoku():
    return generate_sudoku_level("medium")

class UniversalDataset(Dataset):
    def __init__(self, mode='python', size=10000):
        self.mode = mode
        self.data = []
        self.tokenizer = GeneralTokenizer()
        print(f"Generating {size} {mode} samples...")
        for _ in range(size):
            if mode == 'sudoku':
                x, y = generate_sudoku()
            else:
                x, y = "a=1;print(a)", "1" # Placeholder for now
            self.data.append((self.tokenizer.encode(x), self.tokenizer.encode(y)))
    
    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.data[index]