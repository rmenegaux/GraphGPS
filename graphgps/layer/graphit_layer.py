import torch
import torch.nn as nn

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

class GraphiT_Layer(nn.Module):
    """
    In this layer we will define the special multi head attention as in GraphiT.
    
    attributes:
    ----------
        out_dim (int): output dimensionality
        num_heads (int): number of attention heads to use
        use_edge_features (bool): whether or not to use edge features in attention score
        Q (fct): to be used to define the query matrix of the attention
            h [n_batch, num_nodes, in_dim] |-> Q(h) [n_batch, num_nodes, out_dim * num_heads]
        K (fct): to be used to define the key matrix of the attention
            h [n_batch, num_nodes, in_dim] |-> K(h) [n_batch, num_nodes, out_dim * num_heads]
        E (fct): to be used to define the edge embedding of the attention
            h [n_batch, num_edges, in_dim_edges] |-> E(h) [n_batch, num_edges, out_dim * num_heads]
        V (fct): to be used to define the value matrix of the attention
            h [n_batch, num_nodes, in_dim] |-> V(h) [n_batch, num_nodes, out_dim * num_heads]
            
            coef_alpha = softmax(operation(Q, K, E))
            output = V * coef_alpha
        
        attn_dropout (fct): Dropout function to be used on attention scores
        QK_op (str): which operation to use between Q and K mat.
            Either 'multiplication' or 'addition'
        KE_op (str): which operation to use between Q and K mat.
            Either 'multiplication' or 'addition'
        edge_out_dim (None): Amount of edges output dimension to use.
            Any value different than 1, leads to using out_dim.
    """
    def __init__(self, in_dim, in_dim_edges, out_dim, num_heads,
                 use_bias=False, use_edge_features=True,
                 attn_dropout=0.0, QK_op='multiplication', KE_op='addition',
                 edge_out_dim=None):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_edge_features = use_edge_features
        self.QK_op = QK_op
        self.KE_op = KE_op
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        if self.use_edge_features:
            assert (self.QK_op in ['multiplication', 'addition']) and (self.KE_op in ['multiplication', 'addition'])
            edge_out_dim = 1 if (self.QK_op=='multiplication' and self.KE_op=='addition' and edge_out_dim==1) else out_dim
            self.E = nn.Linear(in_dim_edges, edge_out_dim * num_heads, bias=use_bias)

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)

        
    def forward(self, h, edge_features=None, mask=None):
        
        Q = self.Q(h) # [n_batch, num_nodes, out_dim * num_heads]
        K = self.K(h) # [n_batch, num_nodes, out_dim * num_heads]
        V = self.V(h) # [n_batch, num_nodes, out_dim * num_heads]

        n_batch, num_nodes = Q.size()[0:2]

        # Reshaping into [num_heads, num_nodes, feat_dim] to 
        # get projections for multi-head attention
        Q = Q.view(n_batch, num_nodes, self.num_heads, self.out_dim) # [n_batch, num_nodes, num_heads, out_dim]
        K = K.view(n_batch, num_nodes, self.num_heads, self.out_dim) # [n_batch, num_nodes, num_heads, out_dim]
        V = V.view(n_batch, num_nodes, self.num_heads, self.out_dim) # [n_batch, num_nodes, num_heads, out_dim]

        # Normalize by sqrt(head dimension)
        scaling = float(self.out_dim) ** -0.5
        K = K * scaling


        # Automatic combination with argument:
        # - Attention is Q . K + E(multi_dim) :
        #    1. int = einsum('bihk,bjhk->bijh', Q, K)      (bijh)
        #    2. out = int.unsqueeze(-1) + E                (bijh1 + bijhk)
        # - Attention is Q . K + E(scalar) :
        #    1. int = einsum('bihk,bjhk->bijh', Q, K)      (bijh)
        #    2. E from bijhk to bijh by setting self.E to Linear(in_dim_edges, num_heads)
        #    3. out = int + modified_E                      (bijh + bijh)
        # - Attention is Q + K + E :
        #    1. out = Q.unsqueeze(2) + K.unsqueeze(1) + E  (bi1hk + b1jhk + bijhk)
        # - Attention is Q . K . E :
        #    1. out = einsum('bihk,bjhk,bijhk->bijh', Q, K, E)

        if self.use_edge_features:
            E = self.E(edge_features) # [n_batch, num_nodes, num_nodes, out_dim * num_heads]
            E = E.view(n_batch, num_nodes, num_nodes, self.num_heads, -1) # [n_batch, num_nodes, num_nodes, num_heads, out_dim or 1]
            if self.QK_op=='multiplication':
                if self.KE_op=='multiplication': # Attention is Q . K . E
                    scores = torch.einsum('bihk,bjhk,bijhk->bijh', Q, K, E)
                else: # means addition, # Attention is Q . K + E(multi_dim or scalar)
                    scores = torch.einsum('bihk,bjhk->bijh', Q, K).unsqueeze(-1) + E # eventually it ends with dimension of 1
            
            else: # means addition, Attention is Q + K + E
                scores = Q.unsqueeze(2) + K.unsqueeze(1) + E # (bi1hk + b1jhk + bijhk)

        # Apply exponential and clamp for numerical stability
        scores = torch.exp(scores.clamp(-5, 5)) # [n_batch, num_heads, num_nodes, num_nodes]
        # scores = torch.exp(scores - scores.amax(dim=(-2, -1), keepdim=True))

        # Make sure attention scores for padding are 0
        attn_mask = mask.view(-1, num_nodes, 1, 1, 1) * mask.view(-1, 1, num_nodes, 1, 1) # [n_batch, num_nodes, num_nodes, 1, 1]
        if attn_mask is not None:
            scores = scores * attn_mask
            # scores = scores * mask.view(-1, num_nodes, 1, 1, 1) * mask.view(-1, 1, num_nodes, 1, 1)

        # TODO: change that to a full softmax
        # softmax_denom = scores.sum(-1, keepdim=True).clamp(min=1e-6) # [n_batch, num_heads, num_nodes, 1]
        softmax_denom = scores.sum(2).clamp(min=1e-6) # [n_batch, num_heads, num_nodes, out_dim]

        # TODO: 
        # - attn_mask.expand(n_batch, num_nodes, num_nodes, self.num_heads, 1) ??
        # - attn_mask.expand(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim) ??
        attn_mask = self.attn_dropout(attn_mask.float())
        scores = scores * attn_mask

        # h = scores @ V # [n_batch, num_heads, num_nodes, out_dim]
        # TODO: add Edges to V as V*E   h = torch.einsum('bhij,bhjk,bijhk->bhik', scores, V, E)
        # or: h = scores * V * E in which case, other einsum with E after scores * V
        h = torch.einsum('bijhk,bjhk->bihk', scores, V)
        # Normalize scores
        h = h / softmax_denom
        # Concatenate attention heads
        h = h.view(-1, num_nodes, self.num_heads * self.out_dim) # [n_batch, num_nodes, out_dim * num_heads]

        return h