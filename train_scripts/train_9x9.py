import torch
import torch.optim as optim
import torch.nn as nn
from dataset_9x9 import Tokenizer9x9, generate_9x9_level
from model.final_structure import RecursiveTransformer # Use your existing model class!

# --- CONFIG FOR 9x9 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # Smaller batch size because model is bigger
LR = 1e-4

# It takes much longer to learn 9x9
STAGES = [
    ("easy", 5000),   # Learn basic row/col exclusion
    ("medium", 10000),# Learn block interactions
    ("hard", 20000)   # Learn deep backtracking logic
]

tokenizer = Tokenizer9x9()

def train():
    print(f"🔥 Starting 9x9 Training on {DEVICE}...")
    
    # --- BIGGER MODEL ---
    model = RecursiveTransformer(
        vocab_size=20, # 0-9 + separators
        d_model=384,   # Bigger Brain
        num_heads=12, 
        num_layers=6,  # Deeper Brain
        max_len=128,   # Longer Sequence
        dropout=0.1
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding

    for stage, steps in STAGES:
        print(f"\n📚 Stage: {stage.upper()}")
        model.train()
        
        for i in range(steps):
            # Generate Batch
            inputs, targets = [], []
            for _ in range(BATCH_SIZE):
                inp, tgt = generate_9x9_level(stage)
                inputs.append(tokenizer.encode(inp))
                targets.append(tokenizer.encode(tgt))
            
            x = torch.stack(inputs).to(DEVICE)
            y = torch.stack(targets).to(DEVICE)
            
            optimizer.zero_grad()
            
            # Use 24 loops for 9x9 thinking
            all_outs = model(x, loops=24) 
            
            # Loss on final step (focus on the answer)
            # You can also use average loss across steps
            final_step = all_outs[:, -1, :, :]
            loss = criterion(final_step.reshape(-1, 20), y.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"   Batch {i}/{steps} | Loss: {loss.item():.4f}")
        
        torch.save(model.state_dict(), f"brain_9x9_{stage}.pth")

    torch.save(model.state_dict(), "brain_9x9_final.pth")

if __name__ == "__main__":
    train()