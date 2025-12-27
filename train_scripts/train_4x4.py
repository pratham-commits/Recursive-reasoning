import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from dataset_4x4 import GeneralTokenizer, generate_sudoku_level
from model.final_structure import RecursiveTransformer

# --- CONFIG ---
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4

# Curriculum Stages: (Difficulty, Number of Batches)
# We train more on Hard because it's... hard.
# STAGES = [
#     ("easy", 100),   # Warmup: Learn to fill simple gaps
#     ("medium", 200), # Practice: Learn row/col scanning
#     ("hard", 300)    # Mastery: Learn deep recursion
# ]
STAGES = [
    ("easy", 2000),    # ~45 mins
    ("medium", 4000),  # ~1.5 hours
    ("hard", 10000)    # ~3-4 hours (The most important part)
]
tokenizer = GeneralTokenizer()

def augment_puzzle(input_str, target_str):
    """
    Symbol Swapping: Randomly permute the numbers 1-4.
    If 1 becomes 3, and 2 becomes 4... the logic is identical, 
    but the puzzle looks completely new to the model.
    """
    if random.random() > 0.5: # 50% chance to apply augmentation
        mapping = ["1", "2", "3", "4"]
        random.shuffle(mapping) # e.g. ['3', '1', '4', '2']
        
        # Create a translation dictionary
        trans_dict = {str(i+1): mapping[i] for i in range(4)}
        
        # Apply to strings (character by character)
        def apply_map(s):
            res = []
            for char in s:
                if char in trans_dict:
                    res.append(trans_dict[char])
                else:
                    res.append(char)
            return "".join(res)
            
        return apply_map(input_str), apply_map(target_str)
    
    return input_str, target_str
def get_infinite_batch(batch_size, difficulty):
    targets =[]
    inputs=[] 
    for _ in range(batch_size):
        # 1. Generate Fresh Logic
        i_str, t_str = generate_sudoku_level(difficulty)
        
        # 2. Apply Augmentation (Symbol Swap)
        i_str, t_str = augment_puzzle(i_str, t_str)
        
        inputs.append(tokenizer.encode(i_str))
        targets.append(tokenizer.encode(t_str))
        
    return torch.stack(inputs), torch.stack(targets)

def train():
    print(f"🔥 Starting Advanced Training on {DEVICE}...")
    
    # Initialize Model with Dropout
    model = RecursiveTransformer(vocab_size=35, num_layers=4, dropout=0.1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    total_batches = sum(s[1] for s in STAGES)
    current_batch = 0
    
    for stage_name, num_batches in STAGES:
        print(f"\n📚 Entering Curriculum Stage: {stage_name.upper()}")
        
        model.train()
        avg_loss = 0
        
        for i in range(num_batches):
            # Get Infinite Data (No Fixed Dataset!)
            x, y = get_infinite_batch(BATCH_SIZE, stage_name)
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward Pass with 16 Loops
            all_outputs = model(x, loops=16)
            
            # Deep Supervision Loss
            loss = 0
            for step in range(16):
                step_out = all_outputs[:, step, :, :]
                loss += criterion(step_out.reshape(-1, 35), y.reshape(-1))
            loss = loss / 16
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            avg_loss += loss.item()
            current_batch += 1
            
            if i % 10 == 0:
                print(f"   Stage {stage_name} | Batch {i}/{num_batches} | Loss: {loss.item():.4f}")
        
        print(f"✅ Stage {stage_name.upper()} Complete. Avg Loss: {avg_loss/num_batches:.4f}")
        
        # Save checkpoint after every stage
        torch.save(model.state_dict(), f"recursive_brain_{stage_name}.pth")

    # Final Save
    torch.save(model.state_dict(), "recursive_brain.pth")
    print("\n🎓 Graduation Day! Model saved as 'recursive_brain.pth'")

if __name__ == "__main__":
    train()