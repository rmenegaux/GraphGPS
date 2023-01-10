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
                 edge_out_dim=None, dropout_dim='connections'):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_edge_features = use_edge_features
        self.QK_op = QK_op
        self.KE_op = KE_op
        self.dropout_dim = dropout_dim
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        if self.use_edge_features:
            assert (self.QK_op in ['multiplication', 'addition']) and (self.KE_op in ['multiplication', 'addition'])
            self.edge_out_dim = 1 if (self.QK_op=='multiplication' and self.KE_op=='addition' and edge_out_dim==1) else out_dim
            self.E = nn.Linear(in_dim_edges, self.edge_out_dim * num_heads, bias=use_bias)

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

        if self.use_edge_features:
            E = self.E(edge_features) # [n_batch, num_nodes, num_nodes, out_dim * num_heads]
            E = E.view(n_batch, num_nodes, num_nodes, self.num_heads, -1) # [n_batch, num_nodes, num_nodes, num_heads, out_dim or 1]
            if self.QK_op=='multiplication':
                if self.KE_op=='multiplication': # Attention is Q . K . E
                    scores = torch.einsum('bihk,bjhk,bijhk->bijh', Q, K, E).unsqueeze(-1)
                else: # means addition, # Attention is Q . K + E(multi_dim or scalar)
                    scores = torch.einsum('bihk,bjhk->bijh', Q, K).unsqueeze(-1) + E # eventually it ends with dimension of 1
            
            else: # means addition, Attention is Q + K + E
                scores = Q.unsqueeze(2) + K.unsqueeze(1) + E # (bi1hk + b1jhk + bijhk)

            # scores is in bijhk, with k being eventually 1

        # BUG: need to check the dimensionality of `scores` as it depends on the operations we choose
        # Mask to -np.inf such that exp is 0 and you can sum over it as a softmax
        attn_mask = mask.view(-1, num_nodes, 1, 1, 1) * mask.view(-1, 1, num_nodes, 1, 1) # [n_batch, num_nodes, num_nodes, 1, 1]
        scores = torch.nn.functional.softmax(torch.where(attn_mask==0, -float('inf'), scores))
        
        # Then dropout the scores.
        if self.dropout_dim=='feature': # We drop some features for each head
            attn_mask = attn_mask.expand(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim) # to remove only some output connection features for some heads
        elif self.dropout_dim=='head': # We drop some heads for each connection (all their features)
            attn_mask = attn_mask.expand(n_batch, num_nodes, num_nodes, self.num_heads, 1)
        elif self.dropout_dim=='node': # We drop some nodes for each structure (all their connections)
            attn_mask = attn_mask[:,:,0].view(n_batch, num_nodes, 1, 1, 1)
        # Default is: We drop some connections for each node (all their heads)
        scores = scores * self.attn_dropout(attn_mask.float()) # Zeros-out elements along last dimension

        # Compute with Value matrix to finish attention, out size: [n_batch, num_nodes, num_heads, out_dim]
        if self.V_with_edges:
            # h = scores @ (V + E)
            # We must match last dimensions
            if self.QK_op=='multiplication' and self.KE_op=='multiplication':
                # then scores.size[-1] is 1 while for V and E it's out_dim
                equation = 'bijhl,bjhk,bijhk->bihk'
            elif self.edge_out_dim==1:
                # then scores.size[-1] is 1 as well as E, while for V it's out_dim
                equation = 'bijhl,bjhk,bijhl->bihk'
            else:
                equation = 'bijhk,bjhk,bijhk->bihk'
            h = torch.einsum(equation, scores, V, E)
        else: # Standard and default one
            # h = scores @ V
            h = torch.einsum('bijhk,bjhk->bihk', scores, V)
        # Concatenate attention heads
        h = h.view(n_batch, num_nodes, -1) # [n_batch, num_nodes, out_dim * num_heads]

        return h