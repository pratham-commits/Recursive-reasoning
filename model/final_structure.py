import torch
import torch.nn as nn
from .layers import MultiHeadAttention, FeedForward

class RecursiveBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) # Regularization
        
    def forward(self, x):
        x = self.norm1(x)
        # Apply dropout to the output of Attention before adding residual
        x = x + self.dropout(self.attn(x))
        
        x = self.norm2(x)
        # Apply dropout to the output of FeedForward before adding residual
        x = x + self.dropout(self.ff(x))
        return x

class RecursiveTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512, max_len=64, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        self.layers = nn.ModuleList([
            RecursiveBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.input_proj = nn.Linear(d_model * 3, d_model) 
        self.answer_proj = nn.Linear(d_model * 2, d_model)

        self.norm_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, loops=16):
        b, t = input_ids.shape
        x_emb = self.embedding(input_ids) + self.pos_embedding[:, :t, :]
        y_emb = x_emb.clone()           
        z_emb = torch.zeros_like(x_emb) 
        
        all_step_outputs = [] 
        
        for k in range(loops):
            # Step 1: Think (z)
            combined_z = torch.cat([x_emb, y_emb, z_emb], dim=-1)
            curr_z = self.input_proj(combined_z)
            for layer in self.layers:
                curr_z = layer(curr_z)
            z_emb = curr_z 
            
            # Step 2: Answer (y)
            combined_y = torch.cat([y_emb, z_emb], dim=-1)
            curr_y = self.answer_proj(combined_y)
            for layer in self.layers:
                curr_y = layer(curr_y)
            
            logits = self.head(self.norm_final(curr_y))
            all_step_outputs.append(logits)
            
            probs = torch.softmax(logits, dim=-1)
            y_emb = torch.matmul(probs, self.embedding.weight) + self.pos_embedding[:, :t, :]
            
        return torch.stack(all_step_outputs, dim=1)