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
                 use_bias=False, share_edge_features=True,
                 attn_dropout=0.0):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.share_edge_features = share_edge_features
        if not self.share_edge_features:
            self.E_att = nn.Linear(in_dim_edges, out_dim * num_heads, bias=use_bias)
            # self.E2 = nn.Linear(in_dim_edges, 1, bias=use_bias)
            self.E_value = nn.Linear(in_dim_edges, out_dim * num_heads, bias=use_bias)

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)

        
    def forward(self, h, e=None, e_att=None, e_value=None, mask=None):
        
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
        
        scores = torch.einsum('bihk,bjhk->bijh', Q_h, K_h).unsqueeze(-1)
        # scores = Q_h.unsqueeze(1) + K_h.unsqueeze(2)
        E_att = e_att if self.share_edge_features else self.E_att(e)
        E_value = e_value if self.share_edge_features else self.E_value(e)

        scores = scores + E_att.view(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim)

        # Apply exponential and clamp for numerical stability
        scores = torch.exp(scores.clamp(-5, 5)) # [n_batch, num_heads, num_nodes, num_nodes]
        # scores = torch.exp(scores - scores.amax(dim=(-2, -1), keepdim=True))

        # Make sure attention scores for padding are 0
        attn_mask = mask.view(-1, num_nodes, 1, 1, 1) * mask.view(-1, 1, num_nodes, 1, 1)
        if attn_mask is not None:
            scores = scores * attn_mask
            # scores = scores * mask.view(-1, num_nodes, 1, 1, 1) * mask.view(-1, 1, num_nodes, 1, 1)
        
        # softmax_denom = scores.sum(-1, keepdim=True).clamp(min=1e-6) # [n_batch, num_heads, num_nodes, 1]
        softmax_denom = scores.sum(2).clamp(min=1e-6) # [n_batch, num_heads, num_nodes, out_dim]

        # attn_mask = attn_mask.expand(n_batch, num_nodes, num_nodes, self.num_heads, 1)
        attn_mask = self.attn_dropout(attn_mask.float())
        scores = scores * attn_mask

        # h = scores @ V_h # [n_batch, num_heads, num_nodes, out_dim]
        # h = torch.einsum('bhij,bhjk,bijhk->bhik', scores, V_h, E)
        h = torch.einsum('bijhk,bjhk->bihk', scores, V_h)
        h = h + (scores * E_value.view(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim)).sum(2)
        # Normalize scores
        h = h / softmax_denom
        # Concatenate attention heads
        h = h.view(-1, num_nodes, self.num_heads * self.out_dim) # [n_batch, num_nodes, out_dim * num_heads]

        return h
