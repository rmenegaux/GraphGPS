import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.config import cfg

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
                 use_bias=False, use_attention_pe=True, use_edge_features=False,
                 attn_dropout=0.0):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_edge_features = use_edge_features
        self.use_attention_pe = use_attention_pe
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)

        
    def forward(self, h, e_att, e_val, k_RW=None, mask=None, adj=None):
        
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

            # attention(i, j) = sum(Q_i * K_j + E_ij) (multi-dim)
            scores = torch.einsum('bihk,bjhk->bijh', Q_h, K_h).unsqueeze(-1)
            scores = scores + e_att.view(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim) # [n_batch, num_nodes, num_nodes, num_heads, out_dim]
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

        attn_mask = self.attn_dropout(attn_mask.float())
        scores = scores * attn_mask

        # h = scores @ V_h # [n_batch, num_heads, num_nodes, out_dim]
        # h = torch.einsum('bhij,bhjk,bijhk->bhik', scores, V_h, E)
        h = torch.einsum('bijhk,bjhk->bihk', scores, V_h)
        h = h + (scores * e_val.view(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim)).sum(2)
        # Normalize scores
        h = h / softmax_denom
        # Concatenate attention heads
        h = h.view(-1, num_nodes, self.num_heads * self.out_dim) # [n_batch, num_nodes, out_dim * num_heads]

        return h
    

class GraphiT_GT_Layer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, **layer_params):
                #  double_attention=False, dropout=0.0,
                #  layer_norm=False, batch_norm=True, residual=True, use_attention_pe=False,
                #  use_edge_features=True, update_edge_features=False, update_pos_enc=False, use_bias=False
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = layer_params['dropout']
        self.residual = layer_params['residual']
        self.layer_norm = layer_params['layer_norm']     
        self.batch_norm = layer_params['batch_norm']
        self.instance_norm = layer_params['instance_norm']
        self.feedforward = layer_params['feedforward']
        self.update_edge_features = layer_params['update_edge_features']
        self.use_node_pe = layer_params['use_node_pe']
        self.use_attention_pe = layer_params['use_attention_pe']
        self.multi_attention_pe = layer_params['multi_attention_pe']
        self.normalize_degree = layer_params['normalize_degree']
        self.update_pos_enc = layer_params['update_pos_enc']

        attention_params = {
            param: layer_params[param] for param in ['use_bias', 'use_attention_pe', 'use_edge_features', 'attn_dropout']
        }
        # in_dim*2 if positional embeddings are concatenated rather than summed
        in_dim_h = in_dim*2 if (self.use_node_pe == 'concat') else in_dim
        self.attention_h = MultiHeadAttentionLayer(in_dim_h, in_dim, out_dim//num_heads, num_heads, **attention_params)
        self.O_h = nn.Linear(out_dim, out_dim, bias=False)
        
        if self.update_pos_enc:
            self.attention_p = MultiHeadAttentionLayer(in_dim, in_dim, out_dim//num_heads, num_heads, **attention_params)
            self.O_p = nn.Linear(out_dim, out_dim, bias=False)
        
        self.multi_attention_pe = layer_params['multi_attention_pe']
        self.learnable_attention_pe = (self.use_attention_pe and self.multi_attention_pe == 'aggregate')
        if self.learnable_attention_pe:
            attention_pe_dim = layer_params['attention_pe_dim']
            self.coef = nn.Parameter(torch.ones(attention_pe_dim) / attention_pe_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        if self.instance_norm:
            self.instance_norm1_h = nn.InstanceNorm1d(out_dim)
        
        # FFN for h
        if self.feedforward:
            self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2, bias=False)
            self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim, bias=False)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

        if self.instance_norm:
            self.instance_norm2_h = nn.InstanceNorm1d(out_dim)

        if self.update_edge_features:
            self.B1 = nn.Linear(out_dim, out_dim)
            self.B2 = nn.Linear(out_dim, out_dim)
            self.E12 = nn.Linear(out_dim, out_dim)
            if self.layer_norm:
                self.layer_norm_e = nn.LayerNorm(out_dim)
            if self.batch_norm and self.update_edge_features:
                self.batch_norm_e = nn.BatchNorm1d(out_dim)


    def forward_p(self, p, e, k_RW=None, mask=None, adj=None):
        '''
        Update positional encoding p
        '''
        p_in1 = p # for residual connection
    
        p = self.attention_p(p, e, k_RW=k_RW, mask=mask, adj=adj)  
        p = F.dropout(p, self.dropout, training=self.training)
        p = self.O_p(p)
        p = torch.tanh(p)
        if self.residual:
            p = p_in1 + p # residual connection

        return p

    def feed_forward_block(self, h, mask=None):
        '''
        Add dense layers to the self-attention
        '''
        # FFN for h
        h_in2 = h # for second residual connection
        if self.layer_norm:
            h = self.layer_norm2_h(h)
        h = self.FFN_h_layer1(h)
        h = F.gelu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection       
    
        # if self.layer_norm:
        #     h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h.transpose(1,2)).transpose(1,2)
            # h = self.batch_norm2_h(h.transpose(1,2), input_mask=mask.unsqueeze(1)).transpose(1,2)

        if self.instance_norm:
            # h = self.instance_norm2_h(h.transpose(1,2)).transpose(1,2)
            h = self.instance_norm1_h(h)
        return h

    def forward(self, h, p, e_att, e_val, k_RW=None, mask=None, adj=None):

        h_in1 = h # for first residual connection
        
        # [START] For calculation of h -----------------------------------------------------------------
        # h = combine_h_p(h, p, operation=self.use_node_pe)
        # h = h + p

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.learnable_attention_pe:
            # Compute the weighted average of the relative positional encoding
            with torch.no_grad():
                coef = self.coef.data.clamp(min=0)
                coef /= coef.sum(dim=0, keepdim=True)
                self.coef.data.copy_(coef)
            k_RW = torch.tensordot(self.coef, k_RW, dims=[[0], [-1]])
        if self.use_attention_pe and self.multi_attention_pe != 'per_head':
            # Add dimension for the attention heads
            k_RW = k_RW.unsqueeze(1)
        elif self.use_attention_pe and self.multi_attention_pe == 'per_head':
            # One relative attention matrix per attention head
            k_RW = k_RW.transpose(1, -1)

        # multi-head attention out
        h = self.attention_h(h, e_att, e_val, k_RW=k_RW, mask=mask, adj=adj)
        
        if self.update_edge_features: 
            e = self.forward_edges(h_in1, e)
       
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        # Normalize by degree
        # The degree computation could be moved to the DataLoader
        if self.normalize_degree:
            degrees = adj.sum(dim=-1, keepdim=True)
            degrees[degrees == 0] = 1
            h = h * degrees.pow(-0.5)

        if self.residual:
            h = h_in1 + h # residual connection
            
        # if self.layer_norm:
        #     h = self.layer_norm1_h(h)

        if self.batch_norm:
            # Apparently have to do this double transpose for 3D input
            h = self.batch_norm1_h(h.transpose(1,2)).transpose(1,2)
            # h = self.batch_norm1_h(h.transpose(1,2), input_mask=mask.unsqueeze(1)).transpose(1,2)
            # Set padding back to zero
            if mask is not None:
                h = mask.unsqueeze(-1) * h

        if self.instance_norm:
            # h = self.instance_norm1_h(h.transpose(1,2)).transpose(1,2)
            h = self.instance_norm1_h(h)

        if self.feedforward:
            h = self.feed_forward_block(h, mask=mask)
            # Set padding back to zero
            if mask is not None:
                h = mask.unsqueeze(-1) * h
                
        if self.use_node_pe and self.update_pos_enc:
            p = self.forward_p(p, e, k_RW=k_RW, mask=mask, adj=adj)

        return h, p, None
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.gelu(y)
        y = self.FC_layers[self.L](y)
        return y


"""
    GraphiT-GT and GraphiT-GT-LSPE
    
"""


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, full_atom_feature_dims):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.atom_embedding_list[i](x[...,i])

        return x_embedding

class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim, full_bond_feature_dims):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[-1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[...,i])

        return bond_embedding

def global_pooling(x, readout='mean', mask=None):
    if readout == 'mean':
        if mask is not None:
            return x.sum(dim=1) / mask.sum(dim=1)
        else:
            return x.mean(dim=1)
    elif readout == 'max':
        return x.max(dim=1)
    elif readout == 'sum':
        return x.sum(dim=1)
    elif readout == 'first':
        return x[:, 0, :]


@register_network('GraphiTModel')
class GraphiTNet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        num_atom_type = cfg.dataset.node_encoder_num_types
        num_bond_type = cfg.dataset.edge_encoder_num_types
        
        self.use_node_pe = '+' in cfg.dataset.node_encoder_name
        if self.use_node_pe:
            self.pos_enc_dim = cfg.posenc_RWSE.dim_pe
        self.progressive_attention = False
        
        GT_layers = cfg.gt.layers
        GT_hidden_dim = cfg.gt.dim_hidden
        GT_out_dim = GT_hidden_dim
        GT_n_heads = cfg.gt.n_heads
        
        self.readout = 'sum'

        self.n_classes = 1
        in_feat_dropout = 0.0
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layer_norm = cfg.gt.layer_norm
        if self.layer_norm:
            self.layer_norm_h = nn.LayerNorm(GT_out_dim, elementwise_affine=False)

        self.use_edge_features = True

        layer_params = {
            'use_bias': False,
            'dropout': cfg.gnn.dropout,
            'attn_dropout': cfg.gt.attn_dropout,
            'layer_norm': cfg.gt.layer_norm,
            'batch_norm': cfg.gt.batch_norm,
            'instance_norm': False,
            'residual': True,
            'use_node_pe': self.use_node_pe,
            'use_attention_pe': '+' in cfg.dataset.edge_encoder_name,
            'attention_pe_dim': cfg.posenc_RWSE.dim_pe,
            'multi_attention_pe': True,
            'update_edge_features': False,
            'use_edge_features': True,
            'update_pos_enc': False,
            'normalize_degree': False,
            'feedforward': True
        }
        
        if self.use_node_pe:
            self.batch_norm_p = nn.BatchNorm1d(self.pos_enc_dim)
            self.embedding_p = nn.Linear(self.pos_enc_dim, GT_hidden_dim)

        if isinstance(num_atom_type, list):
            self.embedding_h = AtomEncoder(GT_hidden_dim, num_atom_type)
        else:
            self.embedding_h = nn.Embedding(num_atom_type + 1, GT_hidden_dim, padding_idx=0)
        
        if self.use_edge_features:
            if isinstance(num_bond_type, list):
                self.embedding_e = BondEncoder(GT_hidden_dim, num_bond_type)
            else:
                self.embedding_e = nn.Embedding(num_bond_type + 7, GT_hidden_dim//2, padding_idx=0)
                self.batch_norm_e = nn.BatchNorm1d(layer_params['attention_pe_dim'])
                self.positional_embedding_e = nn.Linear(layer_params['attention_pe_dim'], GT_hidden_dim//2, bias=False)
                self.E = nn.Linear(GT_hidden_dim, GT_hidden_dim)
                # self.E2 = nn.Linear(in_dim_edges, 1, bias=use_bias)
                self.E2 = nn.Linear(GT_hidden_dim, GT_hidden_dim)
        
        self.layers = nn.ModuleList([
            GraphiT_GT_Layer(GT_hidden_dim, GT_hidden_dim, GT_n_heads, **layer_params) for _ in range(GT_layers-1)
            ])
        layer_params['use_attention_pe'] = False 
        layer_params['update_edge_features'] = False
        self.layers.append(
            GraphiT_GT_Layer(GT_hidden_dim, GT_out_dim, GT_n_heads, **layer_params)
            )
        
        if self.use_node_pe:
            self.p_out = nn.Linear(GT_out_dim, self.pos_enc_dim)
            self.Whp = nn.Linear(GT_out_dim+self.pos_enc_dim, GT_out_dim)

        self.MLP_layer = MLPReadout(GT_out_dim, self.n_classes)   # 1 out dim when regression problem        
                
        
    # def forward(self, h, p, e, k_RW=None, mask=None):
    def forward(self, batch):
        # h = h.squeeze()
        h = batch.x
        p = batch.node_pe
        e = batch.edge_dense
        k_RW = batch.edge_pe
        mask = batch.mask
        # Node embedding
        h = self.embedding_h(h.squeeze(-1))
        # Binary adjacency matrix (can be used for attention masking)
        adj = (e > 1) # e==0 is padding and e==1 is non-connected node
        # Edge embedding
        if self.use_edge_features:
            e = self.embedding_e(e)
            # Combine edge type and edge positions
            #e = e + self.positional_embedding_e(k_RW)
            # edge_positional_embedding = self.positional_embedding_e(self.batch_norm_e(k_RW))
            edge_positional_embedding = self.positional_embedding_e(k_RW)
            e = torch.cat((e, edge_positional_embedding), dim=-1)
            e_att = self.E2(e)
            e_val = self.E(e)

        h = self.in_feat_dropout(h)
        
        if self.use_node_pe:
            p = self.batch_norm_p(p.transpose(1,2)).transpose(1,2)
            p = self.embedding_p(p)
            p = mask.unsqueeze(-1) * p
            h = h + p

        for i, conv in enumerate(self.layers):
            # Concatenate/Add/Multiply h and p for first layer (or all layers)
            # if (i == 0) or self.update_pos_enc:
            # # if True:
            #     h = combine_h_p(h, p, operation=self.use_node_pe)
            k_RW_i = k_RW[:, :, :, i] if self.progressive_attention else k_RW
            h, p, e = conv(h, p, e_att, e_val, k_RW=k_RW_i, mask=mask, adj=adj)

        if self.use_node_pe:
            p = self.p_out(p)
            # Concat h and p before classification
            # FIXME: hp is not used for now
            hp = self.Whp(torch.cat((h, p), dim=-1))

        # readout
        # h = global_pooling(h, readout=self.readout, mask=mask.unsqueeze(-1))
       # h = h / mask.sum(-1, keepdim=True).float().sqrt()
        if self.layer_norm:
            h = self.layer_norm_h(h)
        
        # return self.MLP_layer(h)
        h = self.MLP_layer(h)
        return global_pooling(h, readout=self.readout, mask=mask.unsqueeze(-1)), batch.y
    

    # def loss(self, scores, targets):

    #     loss = 0 

    #     if self.n_classes == 1:
    #         loss = nn.L1Loss()(scores, targets)
    #     else:
    #         loss = torch.nn.BCEWithLogitsLoss()(scores, targets)
        
    #     return loss
        