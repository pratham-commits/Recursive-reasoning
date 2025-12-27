import torch
# Notice we now import from the folder 'model'
from model.final_structure import RecursiveTransformer

def test_model():
    print("Testing Modular Architecture...")
    
    # Initialize Model
    model = RecursiveTransformer(vocab_size=35)
    
    # Calculate Parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Import Successful.")
    print(f"Total Parameters: {params:,}")
    
    # Test Forward Pass
    dummy_x = torch.randint(0, 35, (2, 10)) # Batch 2, Seq 10
    output = model(dummy_x, loops=16)
    
    print(f"Output Shape: {output.shape}")

if __name__ == "__main__":
    test_model()