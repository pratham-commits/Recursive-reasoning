import numpy as np
import random
import torch

# --- CONFIG ---
# 9x9 Grid = 81 numbers + 9 separators ('|') = 90 tokens roughly
MAX_LEN = 128 

class Tokenizer9x9:
    def __init__(self):
        # Vocab: 0-9, separators, special tokens
        self.vocab = ["<PAD>", "<SOS>", "<EOS>", "|", " "] + [str(i) for i in range(10)]
        self.t2i = {t: i for i, t in enumerate(self.vocab)}
        self.i2t = {i: t for i, t in enumerate(self.vocab)}

    def encode(self, str_grid):
        # Input format: "530070000|600195000|..."
        tokens = ["<SOS>"]
        for char in str_grid:
            if char in self.t2i:
                tokens.append(char)
        tokens.append("<EOS>")
        
        # Padding
        ids = [self.t2i[t] for t in tokens]
        if len(ids) < MAX_LEN:
            ids += [self.t2i["<PAD>"]] * (MAX_LEN - len(ids))
        return torch.tensor(ids[:MAX_LEN], dtype=torch.long)

    def decode(self, ids):
        tokens = []
        for i in ids:
            t = self.i2t.get(int(i), "")
            if t in ["<EOS>", "<PAD>"]: break
            if t != "<SOS>": tokens.append(t)
        return "".join(tokens)

def get_base_9x9():
    """Returns a valid solved 9x9 grid using a pattern shift method."""
    # This generates a valid Latin Square with 3x3 block property
    # Base pattern for a 9x9 grid
    def pattern(r,c): return (3*(r%3) + r//3 + c) % 9
    
    # Shuffle symbols (1-9)
    nums = list(range(1, 10))
    random.shuffle(nums)
    
    # Generate grid
    rows = [g*3 + r for g in random.sample(range(3), 3) for r in random.sample(range(3), 3)] 
    cols = [g*3 + c for g in random.sample(range(3), 3) for c in random.sample(range(3), 3)]
    
    board = [[nums[pattern(r,c)] for c in cols] for r in rows]
    return np.array(board)

def generate_9x9_level(difficulty="medium"):
    base = get_base_9x9()
    puzzle = base.copy()
    
    # How many cells to hide? (81 total)
    if difficulty == "easy": remove = random.randint(30, 35)   # ~50 clues (Very Easy)
    elif difficulty == "medium": remove = random.randint(40, 45) # ~38 clues
    elif difficulty == "hard": remove = random.randint(50, 55)   # ~28 clues
    else: remove = random.randint(58, 62) # Extreme
        
    indices = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(indices)
    
    for i in range(remove):
        r, c = indices[i]
        puzzle[r, c] = 0
        
    # Format: "row1|row2|..."
    inp_str = "|".join(["".join(map(str, r)) for r in puzzle])
    tgt_str = "|".join(["".join(map(str, r)) for r in base])
    
    return inp_str, tgt_str