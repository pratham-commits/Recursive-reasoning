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
    "converts code/sudoku into tensors and vice versa"
    def encode(self,text,max_len = 64):
        for s in ["=", "+", "-", "*", ";", "(", ")", ":", ">", "<", "|"]:
            text = text.replace(s, f" {s} ")
        for d in "0123456789":
            text = text.replace(d, f" {d} ")
        tokens = text.split()
        
        ids = [TOKEN_TO_IDX.get(t,TOKEN_TO_IDX["<PAD>"]) for t in tokens]
        ids = [TOKEN_TO_IDX["<SOS>"]] + ids + [TOKEN_TO_IDX["<EOS>"]]
        
        if len(ids) < max_len:
            ids += [TOKEN_TO_IDX["<PAD>"]]*(max_len-len(ids))
            
        return torch.tensor( ids[:max_len] , dtype=torch.long )
    
    def decode(self,ids):
        tokens = []
        for i in ids:
            t = IDX_TO_TOKEN.get(int(i),"")
            if t not in ["<PAD>","<SOS>","<EOS>"]:
                tokens.append(t)
        return "".join(tokens).replace(" ; ", "; ").replace(" = ", "=").replace(" | ", " | ")
    
def generate_sudoku():
    
    base = np.array([[1, 2, 3, 4],
                     [3, 4, 1, 2],
                     [2, 1, 4, 3],
                     [4, 3, 2, 1]])
    
    if random.random() > 0.5: base[[0, 1]] = base[[1, 0]]
    if random.random() > 0.5: base[[2, 3]] = base[[3, 2]]
    
    problem = base.copy()
    num_missing = random.randint(4,6)
    for _ in range(num_missing):
        r,c = random.randint(0,3), random.randint(0,3)
        problem[r,c] = 0
        
    input_str = " | ".join([" ".join(map(str, row)) for row in problem])
    target_str = " | ".join([" ".join(map(str, row)) for row in base])
    
    return input_str, target_str

def generate_python():
    variables = ['a', 'b', 'c', 'x', 'y', 'z']
    
    # Step 1: Initialize
    v1 = random.choice(variables)
    val1 = random.randint(1, 4)
    current_val = val1
    
    # Step 2: Intermediate Operation
    v2 = random.choice([v for v in variables if v != v1])
    op = random.choice(['+', '*'])
    val2 = random.randint(1, 3)
    
    if op == '+': current_val += val2
    else: current_val *= val2
    
    # Step 3: Final Print
    op2 = random.choice(['+', '-'])
    val3 = random.randint(1, 3)
    
    if op2 == '+': final_res = current_val + val3
    else: final_res = current_val - val3
    
    # "a=2; b=a+3; print(b-1)"
    code = f"{v1}={val1}; {v2}={v1}{op}{val2}; print({v2}{op2}{val3})"
    
    return code, str(final_res)

class UniversalDataset(Dataset):
    def __init__(self, mode = 'python' , size = 10000):
        self.mode = mode
        self.data = []
        self.tokenizer = GeneralTokenizer()
        print(f"Generating {size} {mode} samples...")
        for _ in range(size):
            if mode =='sudoku':
                x,y = generate_sudoku()
            else:
                x,y = generate_python()
                
            self.data.append((self.tokenizer.encode(x), self.tokenizer.encode(y)))
    
    def __len__(self): return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

if __name__ == "__main__":
    print("Testing python generator")
    ds_py = UniversalDataset(mode='python', size=5)
    tok = GeneralTokenizer()
    print(f"Input:  {tok.decode(ds_py[0][0])}")
    print(f"Target: {tok.decode(ds_py[0][1])}")

    # Test Sudoku
    print("\nTesting Sudoku Generator:")
    ds_su = UniversalDataset(mode='sudoku', size=2)
    print(f"Input:  {tok.decode(ds_su[0][0])}")
    print(f"Target: {tok.decode(ds_su[0][1])}")