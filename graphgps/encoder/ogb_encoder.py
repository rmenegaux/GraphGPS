import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                                       register_edge_encoder)

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

@register_node_encoder('Atom1')
class Atom1Encoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(Atom1Encoder, self).__init__()

        self.atom_embedding = torch.nn.Embedding(full_atom_feature_dims[0], emb_dim)
        torch.nn.init.xavier_uniform_(self.atom_embedding.weight.data)


    def forward(self, batch):
        batch.x = self.atom_embedding(batch.x[:, 0])

        return batch

@register_node_encoder('Bond1')
class Bond1Encoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(Bond1Encoder, self).__init__()

        self.bond_embedding = torch.nn.Embedding(full_bond_feature_dims[0], emb_dim)
        torch.nn.init.xavier_uniform_(self.bond_embedding.weight.data)

    def forward(self, batch):
        batch.edge_attr = self.bond_embedding(batch.edge_attr[:, 0])

        return batch
