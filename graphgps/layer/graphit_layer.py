import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch

import numpy as np
import time

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

        
    def forward(self, h, e=None, e_att=None, e_value=None, attn_mask=None):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        # t0 = time.time()

        with CudaTimer('Linears Q, K, V') as _:
            Q_h = self.Q(h) # [n_batch, num_nodes, out_dim * num_heads]
            K_h = self.K(h)
            V_h = self.V(h)

            n_batch = Q_h.size()[0]
            num_nodes = Q_h.size()[1]

            # Reshaping into [num_heads, num_nodes, feat_dim] to 
            # get projections for multi-head attention
            Q_h = Q_h.view(n_batch, num_nodes, self.num_heads, self.out_dim)
            K_h = K_h.view(n_batch, num_nodes, self.num_heads, self.out_dim)
            # V_h = V_h.view(n_batch, num_nodes, self.num_heads, self.out_dim).transpose(1,2)
            V_h = V_h.view(n_batch, num_nodes, self.num_heads, self.out_dim).permute(0, 2, 3, 1)

            # Normalize by sqrt(head dimension)
            scaling = float(self.out_dim) ** -0.5
            K_h = K_h * scaling

        # scores = torch.einsum('bihk,bjhk->bijh', Q_h, K_h).unsqueeze(-1)
        # with CudaTimer('QdotK einsum') as _:
        #     scores = torch.einsum('bihk,bjhk->bijh', Q_h, K_h).unsqueeze(-1)

        # with CudaTimer('QdotK matmul') as _:
        #     scores = torch.matmul(
        #         Q_h.permute(0, 2, 1, 3).contiguous(),
        #         K_h.permute(0, 2, 3, 1).contiguous()
        #     ).unsqueeze(2) # n_batch, num_heads, 1, num_nodes, num_nodes

        with CudaTimer('QdotK matmul') as _:
            scores = torch.matmul(
                Q_h.permute(0, 2, 1, 3),
                K_h.permute(0, 2, 3, 1)
            ).unsqueeze(2) # n_batch, num_heads, 1, num_nodes, num_nodes
        
        # # scores = Q_h.unsqueeze(1) + K_h.unsqueeze(2)
        E_att = e_att if self.share_edge_features else self.E_att(e)
        E_value = e_value if self.share_edge_features else self.E_value(e)

        # with CudaTimer('Adding edge bias 1') as _:
        #     scores1 = scores.expand(-1, -1, self.out_dim, -1, -1).contiguous().view(n_batch, self.num_heads * self.out_dim, num_nodes, num_nodes)
        #     scores1 += E_att.permute(0, 3, 1, 2)


        # with CudaTimer('Attributing attention mask') as _:
        #     attn_mask = torch.zeros((n_batch, 1, num_nodes, num_nodes), dtype=mask.dtype, device=mask.device)

        # with CudaTimer('Masking scores') as _:
        #     # Make sure attention scores for padding are 0
        #     # attn_mask = mask.view(-1, 1, num_nodes, 1) * mask.view(-1, 1, 1, num_nodes)
        #     scores = scores + torch.log(attn_mask).view(-1, 1, 1, num_nodes, num_nodes)

        with CudaTimer('Masking 1e24') as _:
            # Make sure attention scores for padding are 0
            # Keep 1e24 instead of infinity to avoid NaN errors in softmax
            scores = scores - 1e24 * (~attn_mask).view(-1, 1, 1, num_nodes, num_nodes)

        with CudaTimer('Adding edge bias') as _:
            scores = scores + E_att.view(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim).permute(0, 3, 4, 1, 2)
        # with CudaTimer('Reshaping scores') as _:
        #     scores = scores.contiguous().view(n_batch, self.num_heads * self.out_dim, num_nodes, num_nodes)
        # n_batch, num_heads*out_dim, num_nodes, num_nodes

        with CudaTimer('Softmax') as _:
            scores = nn.functional.softmax(scores, dim=-1)

        with CudaTimer('Dropout connections') as _:
            # Dropout connections
            attn_mask = self.attn_dropout(attn_mask.float()).unsqueeze(1).unsqueeze(1)
            scores = scores * attn_mask

        # with CudaTimer('Dropout all') as _:
        #     # Dropout feature wise
        #     scores = self.attn_dropout(scores)

        # with CudaTimer('Softmax manual') as _:
        #     # Apply exponential and clamp for numerical stability
        #     scores = torch.exp(scores.clamp(-5, 5)) # [n_batch, num_heads, num_nodes, num_nodes]
        #     # scores = torch.exp(scores - scores.amax(dim=(-2, -1), keepdim=True))
        # # with CudaTimer('Mask, softmaxDenom') as _:
        # #     # Make sure attention scores for padding are 0
        # #     attn_mask = mask.view(-1, 1, num_nodes, 1) * mask.view(-1, 1, 1, num_nodes)
        # #     if attn_mask is not None:
        # #         scores = scores * attn_mask
        # #         # scores = scores * mask.view(-1, num_nodes, 1, 1, 1) * mask.view(-1, 1, num_nodes, 1, 1)
            
        # #     # softmax_denom = scores.sum(-1, keepdim=True).clamp(min=1e-6) # [n_batch, num_heads, num_nodes, 1]
        #     softmax_denom = scores.sum(-1, keepdim=True).clamp(min=1e-6) # [n_batch, num_heads*out_dim, num_nodes]
        #     scores = scores / softmax_denom
        # #     # attn_mask = attn_mask.expand(n_batch, num_nodes, num_nodes, self.num_heads, 1)
        # #     attn_mask = self.attn_dropout(attn_mask.float())
        # #     scores = scores * attn_mask

        with CudaTimer('scores @ V') as _:
            h = scores @ V_h.unsqueeze(-1) # [n_batch, num_heads, num_nodes, out_dim]
            h = h.squeeze(-1)

        # with CudaTimer('scores @ V contiguous') as _:
        #     h = scores @ V_h.unsqueeze(-1).contiguous()

        # with CudaTimer('scores @ V einsum') as _:
        #     h = torch.einsum('bhij,bhj->bhi', scores, V_h)
        # h = torch.einsum('bhij,bhjk,bijhk->bhik', scores, V_h, E)

        # h = torch.einsum('bijhk,bjhk->bihk', scores, V_h)
        # import pdb; pdb.set_trace()
        # with CudaTimer('scores @ V') as _:
        #     h = torch.matmul(
        #         scores,
        #         V_h.transpose(1, 2).unsqueeze(-1).contiguous()
        #         ).squeeze(-1) # n_batch, num_heads*out_dim, num_nodes


        # h = torch.einsum('bijhk,bjhk->bihk', scores, V_h)
        # torch.matmul(
        #    scores.permute(0, 3, 4, 1, 2).unsqueeze(-2),
        #     E_value.view(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim).permute(0, 3, 4, 1, 2).unsqueeze(-1)
        #    )

        with CudaTimer('Adding edge values') as _:
            #h += (scores * E_value.permute(0, 3, 1, 2)).sum(-1)
            # Match the shape of scores
            E_value = E_value.view(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim).permute(0, 3, 4, 1, 2)
            # Add the edge messages
            h += (scores * E_value).sum(-1)

        # with CudaTimer('Adding edge values einsum') as _:
        #     h = h + torch.einsum('bhij,bijh->bhi', scores, E_value)
        
        # Normalize scores
        # h = h / softmax_denom
        # Concatenate attention heads
        h = h.reshape(n_batch, self.out_dim * self.num_heads, num_nodes).transpose(1, 2)

        # h = h.transpose(1, 2).contiguous() # [n_batch, num_nodes, out_dim * num_heads]
        # h = h.view(n_batch, num_nodes, self.out_dim * self.num_heads)
        # end.record()

        # Waits for everything to finish running
        # torch.cuda.synchronize()
        # torch.cuda.current_stream().synchronize()
        # t1 = time.time()

        # print('all in all {:.2f} ms'.format((t1-t0)*1000))

        # print('all in all ', start.elapsed_time(end))

        # import pdb; pdb.set_trace()
        return h


class CudaTimer2(object):
    def __init__(self, name):
        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
     
    def __enter__(self):
        torch.cuda.current_stream().synchronize()
        self.start.record()
        return None
 
    def __exit__(self, *args):
        self.end.record()
        torch.cuda.current_stream().synchronize()
        print('{:<15}: {:.2f}ms'.format(self.name, self.start.elapsed_time(self.end)))


class CudaTimer(object):
    def __init__(self, name):
        self.name = name
     
    def __enter__(self):
        # self.start = time.time()
        return None
 
    def __exit__(self, *args):
        # torch.cuda.current_stream().synchronize()
        # self.end = time.time()
        # print('{:<30}: {:.2f}ms'.format(self.name, (self.end - self.start)*1000))
        pass