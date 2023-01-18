import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch

import numpy as np

"""
    GraphiT-GT
    
"""
def combine_h_p(h, p, operation='sum'):
    if operation == 'concat':
        h = torch.cat((h, p), dim=-1)
    elif operation == 'sum':
        h = h + p
    elif operation == 'product':
        h = h * p
    return h

"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, in_dim_edges, out_dim, num_heads,
                 use_bias=False, use_attention_pe=True, use_edge_features=True,
                 attn_dropout=0.0):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_edge_features = use_edge_features
        self.use_attention_pe = use_attention_pe
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        if self.use_edge_features:
            self.E = nn.Linear(in_dim_edges, out_dim * num_heads, bias=use_bias)
            # self.E2 = nn.Linear(in_dim_edges, 1, bias=use_bias)
            self.E2 = nn.Linear(in_dim_edges, out_dim * num_heads, bias=use_bias)

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)

        
    def forward(self, h, e=None, mask=None):
        
        Q_h = self.Q(h) # [n_batch, num_nodes, out_dim * num_heads]
        K_h = self.K(h)
        V_h = self.V(h)

        n_batch = Q_h.size()[0]
        num_nodes = Q_h.size()[1]

        # Reshaping into [num_heads, num_nodes, feat_dim] to 
        # get projections for multi-head attention
        Q_h = Q_h.view(n_batch, num_nodes, self.num_heads, self.out_dim)
        K_h = K_h.view(n_batch, num_nodes, self.num_heads, self.out_dim)
        V_h = V_h.view(n_batch, num_nodes, self.num_heads, self.out_dim) # [n_batch, num_heads, num_nodes, out_dim]

        # Normalize by sqrt(head dimension)
        scaling = float(self.out_dim) ** -0.5
        K_h = K_h * scaling

        if self.use_edge_features:
            # E = E.reshape(n_batch, self.num_heads, num_nodes, num_nodes, self.out_dim)
            # E = E.view(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim)

            # E2 = self.E2(e).view(n_batch, 1, num_nodes, num_nodes)

            # attention(i, j) = sum(Q_i * K_j * E_ij)
            # scores = torch.einsum('bihk,bjhk,bijhk->bhij', Q_h, K_h, E)#.unsqueeze(-1)
            # scores = torch.einsum('bihk,bjhk->bijh', Q_h, K_h).unsqueeze(-1)

            # attention(i, j) = sum(Q_i * K_j + E_ij)
            # scores = torch.einsum('bihk,bjhk->bijh', Q_h, K_h)
            # scores = scores + E2 # [n_batch, num_nodes, num_nodes, num_heads, out_dim]

            # attention(i, j) = sum(Q_i * K_j + E_ij) (multi-dim)
            scores = torch.einsum('bihk,bjhk->bijh', Q_h, K_h).unsqueeze(-1)
            # scores = Q_h.unsqueeze(1) + K_h.unsqueeze(2)
            if e is not None:
                E = self.E(e).view(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim)
                E2 = self.E2(e).view(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim)
                scores = scores + E2
                # scores *= float(2) ** -0.5 # [n_batch, num_nodes, num_nodes, num_heads, out_dim]
            else:
                scores = scores.expand(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim)
        else:
            # attention(i, j) = sum(Q_i * K_j)
            scores = torch.einsum('bihk,bjhk->bhij', Q_h, K_h)


        # Apply exponential and clamp for numerical stability
        scores = torch.exp(scores.clamp(-5, 5)) # [n_batch, num_heads, num_nodes, num_nodes]
        # scores = torch.exp(scores - scores.amax(dim=(-2, -1), keepdim=True))

        # Make sure attention scores for padding are 0
        attn_mask = mask.view(-1, num_nodes, 1, 1, 1) * mask.view(-1, 1, num_nodes, 1, 1)
        if attn_mask is not None:
            scores = scores * attn_mask
            # scores = scores * mask.view(-1, num_nodes, 1, 1, 1) * mask.view(-1, 1, num_nodes, 1, 1)

        if self.use_attention_pe:
            pass
            # scores = scores * adj.view(-1, num_nodes, num_nodes, 1, 1) 
            # scores = scores * k_RW
        
        # softmax_denom = scores.sum(-1, keepdim=True).clamp(min=1e-6) # [n_batch, num_heads, num_nodes, 1]
        softmax_denom = scores.sum(2).clamp(min=1e-6) # [n_batch, num_heads, num_nodes, out_dim]

        # attn_mask = attn_mask.expand(n_batch, num_nodes, num_nodes, self.num_heads, 1)
        attn_mask = self.attn_dropout(attn_mask.float())
        scores = scores * attn_mask

        # h = scores @ V_h # [n_batch, num_heads, num_nodes, out_dim]
        # h = torch.einsum('bhij,bhjk,bijhk->bhik', scores, V_h, E)
        h = torch.einsum('bijhk,bjhk->bihk', scores, V_h)
        h = h + (scores * E).sum(2)
        # Normalize scores
        h = h / softmax_denom
        # Concatenate attention heads
        h = h.view(-1, num_nodes, self.num_heads * self.out_dim) # [n_batch, num_nodes, out_dim * num_heads]

        return h
