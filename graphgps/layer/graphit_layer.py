from pyparsing import rest_of_line
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
            output = operation(V, E) * coef_alpha
        
        attn_dropout (fct): Dropout function to be used on attention scores
        QK_op (str): which operation to use between Q and K mat.
            Either 'multiplication' or 'addition'
        KE_op (str): which operation to use between K and E mat.
            Either 'multiplication' or 'addition'
        VE_op (str): which operation to use between V and E mat.
            Either 'multiplication' or 'addition'
        edge_out_dim (None): Amount of edges output dimension to use.
            Any value different than 1, leads to using out_dim.
        dropout_lvl (str): Level at which to apply dropout.
            Either 'feature', 'head', 'node' or else will apply at connection level.
    """
    def __init__(self, in_dim, in_dim_edges, out_dim, num_heads,
                 use_bias=False, use_edge_features=True,
                 attn_dropout=0.0, QK_op='multiplication', KE_op='addition',
                 VE_op=None, edge_out_dim=None, share_edge_features=True,
                 dropout_lvl='connections'):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_edge_features = use_edge_features
        self.QK_op = QK_op
        self.KE_op = KE_op
        self.VE_op = VE_op
        self.dropout_lvl = dropout_lvl
        self.share_edge_features = share_edge_features
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        if self.use_edge_features:
            # assert (self.QK_op in ['multiplication', 'addition']) and (self.KE_op in ['multiplication', 'addition'])
            self.edge_out_dim = 1 if (self.QK_op=='multiplication' and self.KE_op=='addition' and edge_out_dim==1) else out_dim
            if not self.share_edge_features:
                self.E_att = nn.Linear(in_dim_edges, self.edge_out_dim * num_heads, bias=use_bias)
                # E_value will always be multi
                self.E_value = nn.Linear(in_dim_edges, out_dim * num_heads, bias=use_bias)

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)

    def forward(self, h, edge_features=None, e_att=None, e_value=None, attn_mask=None):

        Q = self.Q(h)  # [n_batch, num_nodes, out_dim * num_heads]
        K = self.K(h)  # [n_batch, num_nodes, out_dim * num_heads]
        V = self.V(h)  # [n_batch, num_nodes, out_dim * num_heads]

        n_batch, num_nodes = Q.size()[0:2]

        # Reshaping into [num_heads, num_nodes, feat_dim] to 
        # get projections for multi-head attention
        Q = Q.view(n_batch, num_nodes, self.num_heads, self.out_dim)  # [n_batch, num_nodes, num_heads, out_dim]
        K = K.view(n_batch, num_nodes, self.num_heads, self.out_dim)  # [n_batch, num_nodes, num_heads, out_dim]
        V = V.view(n_batch, num_nodes, self.num_heads, self.out_dim)  # [n_batch, num_nodes, num_heads, out_dim]

        # Normalize by sqrt(head dimension)
        scaling = float(self.out_dim) ** -0.5
        K = K * scaling
        # Q = Q * scaling # must be uncommented for DoubleScaling

        if self.use_edge_features:
            E_att = e_att if self.share_edge_features else self.E_att(edge_features)  # [n_batch, num_nodes, num_nodes, out_dim * num_heads]
            E_att = E_att.view(n_batch, num_nodes, num_nodes, self.num_heads, -1)  # [n_batch, num_nodes, num_nodes, num_heads, out_dim or 1]
            if self.QK_op == 'multiplication':
                if self.KE_op == 'multiplication':  # Attention is Q . K . E
                    scores = torch.einsum('bihk,bjhk,bijhk->bijh', Q, K, E_att).unsqueeze(-1)
                else: # means addition, # Attention is Q . K + E(multi_dim or scalar)
                    # scores = torch.einsum('bihk,bjhk->bijh', Q, K).unsqueeze(-1) + E  # eventually it ends with dimension of 1
                    scores = torch.matmul(
                        Q.permute(0, 2, 1, 3).contiguous(),  # bhik
                        K.permute(0, 2, 3, 1).contiguous()   # bhkj
                    ).permute(0, 2, 3, 1).unsqueeze(-1)      # bhij -> bijh
                    scores = scores + E_att

            elif self.QK_op is None:
                scores = E_att

            else: # means addition, Attention is Q + K + E
                scores = Q.unsqueeze(2) + K.unsqueeze(1) + E_att  # (bi1hk + b1jhk + bijhk)

            # scores is in bijhk, with k being eventually 1
        else:
            scores = torch.einsum('bihk,bjhk->bijh', Q, K).unsqueeze(-1)

        # Mask to -np.inf such that exp is 0 and you can sum over it as a softmax
        # attn_mask = mask.view(-1, num_nodes, 1, 1, 1) * mask.view(-1, 1, num_nodes, 1, 1)  # [n_batch, num_nodes, num_nodes, 1, 1]
        # scores = torch.sparse.softmax((attn_mask * scores).to_sparse(), dim=2).to_dense()
        # scores = torch.nn.functional.softmax(torch.where(attn_mask == 0, -float('inf'), scores.double()), dim=2)
        # scores = torch.where(scores.isnan(), 0., scores)
        # scores = torch.exp(scores.clamp(-5, 5)) * attn_mask
        scores = scores - 1e24 * (~attn_mask)
        # The head dimension is not needed anymore, merge it with the feature dimension
        # scores = scores.reshape(n_batch, num_nodes, num_nodes, -1)
        scores = nn.functional.softmax(scores, dim=2)

        # Then dropout the scores.
        if self.dropout_lvl == 'feature':  # We drop some features for each head
            attn_mask = attn_mask.expand(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim) # to remove only some output connection features for some heads
        elif self.dropout_lvl == 'head':  # We drop some heads for each connection (all their features)
            attn_mask = attn_mask.expand(n_batch, num_nodes, num_nodes, self.num_heads, 1)
        elif self.dropout_lvl == 'node':  # We drop some nodes for each structure (all their connections)
            attn_mask = attn_mask[:,:,0].view(n_batch, num_nodes, 1, 1, 1)
        # Default is: We drop some connections for each node (all their heads)
        scores = scores * self.attn_dropout(attn_mask.float())  # Zeros-out elements along last dimension

        #V = V.double()
        # Compute with Value matrix to finish attention, out size: [n_batch, num_nodes, num_heads, out_dim]
        if self.VE_op is not None:
            E_value = e_value if self.share_edge_features else self.E_value(edge_features) #.double()
            E_value = E_value.view(n_batch, num_nodes, num_nodes, self.num_heads, -1)  # [n_batch, num_nodes, num_nodes, num_heads, out_dim or 1]
            if self.VE_op == 'addition':
                # h = scores @ (V + E)
                # h = torch.einsum('bijhk,bjhk->bihk', scores, V)
                h = torch.matmul(
                        scores.permute(0, 3, 4, 1, 2).contiguous(),
                        V.permute(0, 2, 3, 1).unsqueeze(-1).contiguous()
                    ).squeeze(-1).permute(0, 3, 1, 2)
                if self.edge_out_dim==1:
                    h += torch.einsum('bijhl,bijhk->bihk', scores, E_value)
                else:
                    # h += torch.einsum('bijhk,bijhk->bihk', scores, E_value)
                    h += (scores * E_value).sum(2)

            elif self.VE_op == 'multiplication':
                # h = scores * V * E
                # We must match last dimensions
                if self.QK_op == 'multiplication' and self.KE_op == 'multiplication':
                    # then scores.size[-1] is 1 while for V and E it's out_dim
                    equation = 'bijhl,bjhk,bijhk->bihk'
                elif self.edge_out_dim == 1:
                    # then scores.size[-1] is 1 as well as E, while for V it's out_dim
                    equation = 'bijhl,bjhk,bijhl->bihk'
                else:
                    equation = 'bijhk,bjhk,bijhk->bihk'
                h = torch.einsum(equation, scores, V, E_value)
        else:  # Standard and default one
            # h = scores @ V
            if self.QK_op == 'multiplication' and self.KE_op == 'multiplication':
                h = torch.einsum('bijhl,bjhk->bihk', scores, V)
            else:
                h = torch.einsum('bijhk,bjhk->bihk', scores, V)
        
        # Concatenate attention heads
        try: # FIXME
            h = h.view(n_batch, num_nodes, -1).float()  # [n_batch, num_nodes, out_dim * num_heads]
        except:
            h = h.reshape(n_batch, num_nodes, -1).float()
        
        return h