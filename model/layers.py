import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Equation: Softmax(QK^T / sqrt(d_k))V
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must me divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
        
    def forward(self,z, mask=None):
        batch_size = z.size(0)
        
        Q = self.W_q(z).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(z).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(z).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0 , -1e9)
            
        attn_probs = F.softmax(scores, dim=-1)
        #Weighted Sum
        context = torch.matmul(attn_probs,V)
        #Concatenate Heads & Final Projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model,d_ff), nn.GELU(), nn.Linear(d_ff,d_model))
    
    def forward(self,z):
        return self.net(z)
    
   

        