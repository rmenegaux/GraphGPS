import torch
from torch_scatter import scatter
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

"""
Generic Node and Edge encoders for datasets with node/edge features that
consist of only one type dictionary thus require a single nn.Embedding layer.

The number of possible Node and Edge types must be set by cfg options:
1) cfg.dataset.node_encoder_num_types
2) cfg.dataset.edge_encoder_num_types

In case of a more complex feature set, use a data-specific encoder.

These generic encoders can be used e.g. for:
* ZINC
cfg.dataset.node_encoder_num_types: 28
cfg.dataset.edge_encoder_num_types: 4

* AQSOL
cfg.dataset.node_encoder_num_types: 65
cfg.dataset.edge_encoder_num_types: 5


=== Description of the ZINC dataset === 
https://github.com/graphdeeplearning/benchmarking-gnns/issues/42
The node labels are atom types and the edge labels atom bond types.

Node labels:
'C': 0
'O': 1
'N': 2
'F': 3
'C H1': 4
'S': 5
'Cl': 6
'O -': 7
'N H1 +': 8
'Br': 9
'N H3 +': 10
'N H2 +': 11
'N +': 12
'N -': 13
'S -': 14
'I': 15
'P': 16
'O H1 +': 17
'N H1 -': 18
'O +': 19
'S +': 20
'P H1': 21
'P H2': 22
'C H2 -': 23
'P +': 24
'S H1 +': 25
'C H1 -': 26
'P H1 +': 27

Edge labels:
'NONE': 0
'SINGLE': 1
'DOUBLE': 2
'TRIPLE': 3


=== Description of the AQSOL dataset === 
Node labels: 
'Br': 0, 'C': 1, 'N': 2, 'O': 3, 'Cl': 4, 'Zn': 5, 'F': 6, 'P': 7, 'S': 8, 'Na': 9, 'Al': 10,
'Si': 11, 'Mo': 12, 'Ca': 13, 'W': 14, 'Pb': 15, 'B': 16, 'V': 17, 'Co': 18, 'Mg': 19, 'Bi': 20, 'Fe': 21,
'Ba': 22, 'K': 23, 'Ti': 24, 'Sn': 25, 'Cd': 26, 'I': 27, 'Re': 28, 'Sr': 29, 'H': 30, 'Cu': 31, 'Ni': 32,
'Lu': 33, 'Pr': 34, 'Te': 35, 'Ce': 36, 'Nd': 37, 'Gd': 38, 'Zr': 39, 'Mn': 40, 'As': 41, 'Hg': 42, 'Sb':
43, 'Cr': 44, 'Se': 45, 'La': 46, 'Dy': 47, 'Y': 48, 'Pd': 49, 'Ag': 50, 'In': 51, 'Li': 52, 'Rh': 53,
'Nb': 54, 'Hf': 55, 'Cs': 56, 'Ru': 57, 'Au': 58, 'Sm': 59, 'Ta': 60, 'Pt': 61, 'Ir': 62, 'Be': 63, 'Ge': 64
    
Edge labels: 
'NONE': 0, 'SINGLE': 1, 'DOUBLE': 2, 'AROMATIC': 3, 'TRIPLE': 4
"""




@register_edge_encoder('RWSEEdgeDebug')
class RWSEEdgeEncoderDebug(torch.nn.Module):
    def __init__(self, emb_dim, dense=False):
        super().__init__()

        edge_pe_in_dim = len(cfg.posenc_RWSE.kernel.times) # Size of the kernel-based PE embedding
        self.add_dense_edge_features = dense

        #self.encoder = torch.nn.Linear(edge_pe_in_dim, emb_dim)
        self.emb_dim = emb_dim
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def encoder(self, rwse):
        dims = [s for s in rwse.size()]
        if dims[-1] <= self.emb_dim:
            dims[-1] = self.emb_dim - dims[-1]
            return torch.cat([rwse, rwse.new_zeros(dims)], dim=-1)
        else:
            return rwse[..., :self.emb_dim]

    def forward(self, batch):
        '''
        Ideally this step should be done in the data loading, but the torch geometric DataLoader
        forces us to flatten the dense edge feature matrices
        '''
        # First reshape the edge features into (n_batch, max_nodes, max_nodes, edge_dim)
        batched_edge_features = reshape_flattened_adj(batch.edge_RWSE, batch.batch)
        del batch.edge_RWSE # This is the largest tensor in the batch, deleting it to save space?
        batched_edge_features = self.encoder(batched_edge_features)
        # For the sparse edge_attr we keep the original edges and do not add new ones
        batch_idx, row, col = get_dense_indices_from_sparse(batch.edge_index, batch.batch)
        batch.edge_attr = batched_edge_features[batch_idx, row, col]
        if self.add_dense_edge_features:
            # Maybe directly concatenate this instead, as for NodePE?
            batch.edge_dense = batched_edge_features

        return batch




def get_dense_indices_from_sparse(edge_index, batch):
    batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    return idx0, idx1, idx2


def reshape_flattened_adj(edge_features, batch):
    '''
    Transforms a PyG batch of edge features of shape (N_1+...+N_B, edge_dim) into a 
    padded tensor of shape (B, N_max, edge_dim),
    where N_i = (number of nodes of graph i)^2, and N_max = max(N_i)
    '''
    edge_features_list = unbatch(edge_features, batch)
    num_nodes = [int(e.size(0)**0.5) for e in edge_features_list]
    max_nodes = max(num_nodes)
    n_batch = len(edge_features_list)
    padded_edge_features = edge_features_list[0].new_zeros((n_batch, max_nodes, max_nodes, edge_features_list[0].size(-1)))
    for i, e in enumerate(edge_features_list):
        padded_edge_features[i, :num_nodes[i], :num_nodes[i]] = e.view(num_nodes[i], num_nodes[i], -1)

    return padded_edge_features

# -------------------------
# Copied from latest version of torch geometric
# -------------------------
from typing import List

import torch
from torch import Tensor

from torch_geometric.utils import degree


def unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.
    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)
    :rtype: :class:`List[Tensor]`
    Example:
        >>> src = torch.arange(7)
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    sizes = (degree(batch, dtype=torch.long)**2).tolist()
    return src.split(sizes, dim)
