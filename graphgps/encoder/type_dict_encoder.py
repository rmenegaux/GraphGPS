import torch
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch, to_dense_adj, add_self_loops
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


@register_node_encoder('TypeDictNode')
class TypeDictNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        num_types = cfg.dataset.node_encoder_num_types
        if num_types < 1:
            raise ValueError(f"Invalid 'node_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        batch.x = self.encoder(batch.x[:, 0])

        return batch


@register_edge_encoder('TypeDictEdge')
class TypeDictEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        num_types = cfg.dataset.edge_encoder_num_types
        if cfg.dataset.rings == True and cfg.dataset.rings_coalesce_edges == True:
            # Double each edge type: part of a ring or not
            num_types += cfg.dataset.edge_encoder_num_types
        if num_types < 1:
            raise ValueError(f"Invalid 'edge_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr)

        return batch


def get_rw_landing_probs(ksteps, dense_adj,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    num_nodes = dense_adj.size(1)
    deg = dense_adj.sum(2, keepdim=True) # Out degrees. (batch_size) x (Num nodes) x (1)
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    # P = D^-1 * A
    P = deg_inv * dense_adj  # (Batch_size) x (Num nodes) x (Num nodes)

    rws = []
    rws_all = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            rws_all.append(Pk)
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.stack(rws, dim=2)  # (Batch_size) x (Num nodes) x (K steps)
    rw_landing_all = torch.stack(rws_all, dim=3)  # (Batch_size) x (Num nodes) x (Num nodes) x (K steps)

    return rw_landing, rw_landing_all

@register_edge_encoder('RWSEonthefly')
class RWSEcomputer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        kernel_param = cfg.posenc_RWSE.kernel
        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RWSE")
        self.ksteps = kernel_param.times

    def forward(self, batch):
        dense_adj = to_dense_adj(batch.edge_index, batch=batch.batch)
        # This next line is just to get the node mask (perhaps overkill)
        _, mask = to_dense_batch(batch.edge_index.new_zeros(batch.num_nodes), batch=batch.batch)
        rw_landing, rw_landing_all = get_rw_landing_probs(ksteps=self.ksteps,
                                        dense_adj=dense_adj)
        batch.pestat_RWSE = rw_landing[mask]
        batch.edge_RWSE = rw_landing_all

        return batch

@register_edge_encoder('SPDEdge')
class SPDEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, dense=False):
        super().__init__()

        self.add_dense_edge_features = dense

        num_types = cfg.dataset.spd_max_length + 2
        if num_types < 1:
            raise ValueError(f"Invalid 'spd_max_length': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Shifting lengths by 1 and adding 0s on the diagonal to distinguish
        # non-connected nodes from self-connections
        batch.spd_index, batch.spd_lengths = add_self_loops(
            batch.spd_index, batch.spd_lengths + 1, fill_value=0)
        # Doing things in this order (first embedding, then transforming to dense,
        # ensures that padding remains 0)
        spd_embedding = self.encoder(batch.spd_lengths)
        spd_dense = to_dense_adj(batch.spd_index, batch=batch.batch, edge_attr=spd_embedding)

        batch_idx, row, col = get_dense_indices_from_sparse(batch.edge_index, batch.batch)
        batch.edge_attr = spd_dense[batch_idx, row, col]

        if self.add_dense_edge_features:
            # Maybe directly concatenate this instead, as for NodePE?
            batch.edge_dense = spd_dense

        return batch


@register_edge_encoder('RWSEEdge')
class RWSEEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, dense=False):
        super().__init__()

        edge_pe_in_dim = len(cfg.posenc_RWSE.kernel.times) # Size of the kernel-based PE embedding
        self.reshape = (cfg.posenc_RWSE.precompute == True)
        self.add_dense_edge_features = dense

        self.encoder = torch.nn.Linear(edge_pe_in_dim, emb_dim) # Watch out for padding here? Use bias=False?
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        '''
        Ideally this step should be done in the data loading, but the torch geometric DataLoader
        forces us to flatten the dense edge feature matrices
        '''
        # First reshape the edge features into (n_batch, max_nodes, max_nodes, edge_dim)
        if self.reshape:
            batched_edge_features = reshape_flattened_adj(batch.edge_RWSE, batch.batch)
        else:
            batched_edge_features = batch.edge_RWSE
        del batch.edge_RWSE # This is the largest tensor in the batch, deleting it to save space?
        batched_edge_features = self.encoder(batched_edge_features)
        # For the sparse edge_attr we keep the original edges and do not add new ones
        batch_idx, row, col = get_dense_indices_from_sparse(batch.edge_index, batch.batch)
        batch.edge_attr = batched_edge_features[batch_idx, row, col]
        if self.add_dense_edge_features:
            # Maybe directly concatenate this instead, as for NodePE?
            batch.edge_dense = batched_edge_features

        return batch

@register_edge_encoder('DenseEdge')
class DenseEdgeEncoder(torch.nn.Module):
    '''
    Creates dense edge features `batch.edge_dense` of size 
    `(n_batch, batch_nodes, batch_nodes, emb_dim)` from `batch.edge_attr`

    Fills missing edge features by adding a learnable vector of size `emb_dim`
    to every pair of disconnected nodes (i, j), and another one to the diagonal (i, i)

    `input_batch.edge_attr` should be of compatible last dimension `emb_dim`
    '''
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=3, embedding_dim=emb_dim, padding_idx=0) 
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        '''
        Create a dense edge features matrix `batch.edge_dense`,
        E_ij = edge_attr[i, j] if (i, j) are neighbors
               embedding_1 if i=j
               embedding_2 else
        '''
        if not hasattr(batch, 'edge_dense'):
            batch.edge_dense = to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr)
        A_dense = get_dense_edge_types(batch)
        batch.edge_dense += self.encoder(A_dense)

        return batch

@register_edge_encoder('RingEdge')
class RingEdgeEncoder(torch.nn.Module):
    '''
    Transforms dense edge features by adding a vector to every pair of nodes (i, j)
    that are in a ring
    '''
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=2, embedding_dim=emb_dim, padding_idx=0) 
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        '''
        '''
        # ring_attr = torch.ones_like(batch.ring_index[0])
        ring_dense = to_dense_adj(batch.ring_index, batch=batch.batch).long()
        # FIXME: For sparse version:
        # Should check that ring_index is indexed in the same manner as edge_index after batching
        # Miraculously seems to be the case.
        # ring_index = set(batch.ring_index)
        # edge_mask = torch.Tensor([edge in ring_index for edge in batch.edge_index],
        #                          dtype=batch.edge_index.dtype, device=batch.edge_index.device)
        # batch.edge_attr += self.encoder(edge_mask)

        batch.edge_dense += self.encoder(ring_dense)

        return batch

@register_edge_encoder('Bond1')
class Bond1Encoder(torch.nn.Module):
    '''
    Embed only the first feature in the OGB bond features
    '''
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=2, embedding_dim=emb_dim, padding_idx=0) 
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        '''
        '''
        # ring_attr = torch.ones_like(batch.ring_index[0])
        ring_dense = to_dense_adj(batch.ring_index, batch=batch.batch).long()
        # FIXME: For sparse version:
        # Should check that ring_index is indexed in the same manner as edge_index after batching
        # Miraculously seems to be the case.
        # ring_index = set(batch.ring_index)
        # edge_mask = torch.Tensor([edge in ring_index for edge in batch.edge_index],
        #                          dtype=batch.edge_index.dtype, device=batch.edge_index.device)
        # batch.edge_attr += self.encoder(edge_mask)

        batch.edge_dense += self.encoder(ring_dense)

        return batch

# OGB ring and OGB Ring RWSE
@register_edge_encoder('TypeDictEdge+RWSEEdge')
class TypeRWSEEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.type_encoder = TypeDictEdgeEncoder(emb_dim//2)
        self.pe_encoder = RWSEEdgeEncoder(emb_dim//2)

    def forward(self, batch):
        batch = self.type_encoder(batch)
        edge_types_dense = batch.edge_dense
        edge_types_sparse = batch.edge_attr
        batch = self.pe_encoder(batch)
        batch.edge_dense = torch.cat([edge_types_dense, batch.edge_dense], dim=-1)
        batch.edge_attr = torch.cat([edge_types_sparse, batch.edge_attr], dim=-1)

        return batch

def get_dense_edge_types(batch):
    '''
    Returns a dense complementary adjacency matrix of `batch`,
    of size `(n_batch, batch_nodes, batch_nodes, 1)`
    Matrix A_ij = 0 if (i,j) are connected;
                  1 if i=j;
                  2 otherwise
    FIXME: Should differentiate padding and edges (both have same value as 
           pyG `to_dense_adj` returns 0 for both)
    '''
    edge_attr = 2 * torch.ones_like(batch.edge_index[0])
    edge_index, edge_attr = add_self_loops(batch.edge_index, edge_attr, fill_value=1)
    # to_dense_adj returns a float tensor even when edge_attr is int...
    A_dense = 2 - to_dense_adj(edge_index, batch=batch.batch, edge_attr=edge_attr).long()

    return A_dense


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
        >>> src = torch.arange(14)
        >>> batch = torch.tensor([0, 0, 0, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]), tensor([9]), tensor([10, 11, 12, 13]))
    """
    sizes = (degree(batch, dtype=torch.long)**2).tolist()
    return src.split(sizes, dim)
