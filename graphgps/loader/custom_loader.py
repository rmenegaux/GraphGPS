from collections.abc import Sequence
from tqdm import tqdm
from typing import Union, List
import numpy as np
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.data import BaseData

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


class CustomDataset(object):
    '''
    Simplified version of a Dataset, with the argument `data_list` containing
    Data objects. Made to avoid the collating (and hence duplicating) of data into
    one big Data() object in the torch geometric InMemoryDataset object.
    Replicates most functionalities necessary to the GraphGym framework
    '''
    def __init__(self, pyg_dataset, transform=None):
        # Extract data from dataset in list form
        self.data_list = [pyg_dataset[i] for i in tqdm(range(len(pyg_dataset)),
                  mininterval=10,
                  miniters=len(pyg_dataset)//20)]
        # self.data_list = list(pyg_dataset)
        # self.data is maintained to withhold information such as split indices
        self.data, self.slices = Data(), None
        self.transform = transform

        # These are to ensure compatibility with the `log_loaded_dataset` function
        if hasattr(pyg_dataset.data, 'num_nodes'):
            self.data.num_nodes = pyg_dataset.data.num_nodes
        elif hasattr(pyg_dataset.data, 'x'):
            self.data.num_nodes = pyg_dataset.data.x.size(0)

        self.num_node_features = pyg_dataset.num_node_features
        self.num_edge_features = pyg_dataset.num_edge_features
        if hasattr(pyg_dataset, 'num_tasks'):
            self.num_tasks = pyg_dataset.num_tasks
        if hasattr(pyg_dataset.data, 'y'):
            self.data.y = pyg_dataset.data.y
        # self.num_classes = pyg_dataset.num_classes

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ) -> Union[List, BaseData]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices."""
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = copy.copy(self.data_list[idx])
            data = data if self.transform is None else self.transform(data)
            return data

        else:
            return self.index_select(idx)

    def __len__(self):
        return len(self.data_list)

    def index_select(self, idx: IndexType) -> List:
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.
        Returns a list of Data"""
        indices = list(range(len(self)))

        if isinstance(idx, slice):
            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        return [self.data_list[i] for i in indices]


import torch
from torch import Tensor
from collections.abc import Sequence
from tqdm import tqdm
import copy

import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from typing import Union, List
import numpy as np

from torch_geometric.transforms import ToDense
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData

def CustomBatch(object):
    def __init__(self):
        self._add_rings = False
    
    def from_list(self, data_list):
        '''
        Initialize batch from list of torch geometric Data objects
        '''
        # with profiler.record_function("COLLATE FUNCTION"):
        batch = list(batch)
        max_len = max(len(g.x) for g in batch)
        dense_transform = ToDense(max_len)
        input_size = batch[0].x.shape[1]
        edge_input_size = 1 if batch[0].edge_attr.dim() == 1 else batch[0].edge_attr.shape[1]

        self.x = batch[0].x.new_zeros((len(batch), max_len, input_size))
        self.edge_dense = batch[0].edge_attr.new_zeros((len(batch), max_len, max_len, edge_input_size), dtype=int).squeeze()
        self.mask = batch[0].edge_index.new_zeros((len(batch), max_len), dtype=bool)
        labels = []
        attention_pe = None
        padded_p = None
        if self.use_node_pe:
            padded_p = torch.zeros((len(batch), max_len, self.node_pe_dimension), dtype=float)
        if self.use_attention_pe:
            attention_pe = torch.zeros((len(batch), max_len, max_len, self.attention_pe_dim)).squeeze()

        for i, g in enumerate(batch):
            labels.append(g.y.view(-1))
            num_nodes = len(g.x)
            # FIXME: Adding 1 to the atom type here, to differentiate between atom 0 and padding
            g.x = g.x + 1
            # edge_index = utils.add_self_loops(batch[i].edge_index, None, num_nodes =  max_len)[0]
            g = dense_transform(g)
            padded_x[i] = g.x

            # adj = utils.to_dense_adj(edge_index).squeeze()
            # FIXME: Adding 1 to the edge type here, to differentiate between padding and non-neighbors
            padded_adj[i, :num_nodes, :num_nodes] = g.adj[:num_nodes, :num_nodes] + 1
            # FIXME: Creating new edge type for ring connections (hard coded for 4 edge types)
            if self._add_rings == True:
                padded_adj[i, :num_nodes, :num_nodes] += 4 * g.ring_adj
            # Adding a special edge type (2) for the diagonal
            padded_adj[i, :num_nodes, :num_nodes] += (g.adj[:num_nodes, :num_nodes] > 0)
            padded_adj[i, :num_nodes, :num_nodes].fill_diagonal_(2)

            mask[i] = g.mask
            if self.use_node_pe:
                padded_p[i, :num_nodes] = g.node_pe
            if self.use_attention_pe:
                attention_pe[i, :num_nodes, :num_nodes] = g.attention_pe
        self.y = default_collate(labels)
        return padded_x, padded_adj, padded_p, mask, attention_pe, default_collate(labels)


class GraphDataset(object):
    def __init__(self, dataset):
        """a pytorch geometric dataset as input
        """
        self.dataset = list(dataset)

        self.use_node_pe = False
        self.use_attention_pe = False
        self._add_rings = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def add_rings(self, max_k=17):
        for g in self.dataset:
            g.rings = get_rings(g.edge_index, max_k=max_k)
            g.ring_adj = torch.zeros((len(g.x), len(g.x)), dtype=int)
            for ring in g.rings:
                for i in ring:
                    for j in ring:
                        g.ring_adj[i, j] = 1
                        g.ring_adj[j, i] = 1
        self._add_rings = True


    def compute_node_pe(self, node_pe, standardize=True, update_stats=True):
        '''
        Add node positional embeddings to the graphs' data.
        Takes as argument a function returning a nodewise positional embedding from a graph
        '''
        for g in self.dataset:
            g.node_pe = node_pe(g, update_stats=update_stats)
        if standardize:
            mean, std = node_pe.get_statistics()
            for g in self.dataset:
                g.node_pe = (g.node_pe - mean) / std.clamp(min=1e-6)
        self.use_node_pe = True
        self.node_pe_dimension = node_pe.get_embedding_dimension()

    def compute_attention_pe(self, attention_pe, standardize=True, update_stats=True):
        '''
        Takes as argument a function returning an edgewise positional embedding from a graph
        '''
        for g in self.dataset:
            g.attention_pe = attention_pe(g, update_stats=update_stats)
        if standardize:
            for g in self.dataset:
                g.attention_pe = attention_pe.standardize(g.attention_pe)
        self.use_attention_pe = True
        self.attention_pe_dim = attention_pe.get_dimension()

    def collate_fn(self):
        def collate(batch):
            # with profiler.record_function("COLLATE FUNCTION"):
            batch = list(batch)
            max_len = max(len(g.x) for g in batch)
            dense_transform = ToDense(max_len)
            input_size = batch[0].x.shape[1]
            edge_input_size = 1 if batch[0].edge_attr.dim() == 1 else batch[0].edge_attr.shape[1]

            padded_x = torch.zeros((len(batch), max_len, input_size), dtype=int)
            padded_adj = torch.zeros((len(batch), max_len, max_len, edge_input_size), dtype=int).squeeze()
            mask = torch.zeros((len(batch), max_len), dtype=bool)
            labels = []
            attention_pe = None
            padded_p = None
            if self.use_node_pe:
                padded_p = torch.zeros((len(batch), max_len, self.node_pe_dimension), dtype=float)
            if self.use_attention_pe:
                attention_pe = torch.zeros((len(batch), max_len, max_len, self.attention_pe_dim)).squeeze()

            for i, g in enumerate(batch):
                labels.append(g.y.view(-1))
                num_nodes = len(g.x)
                # FIXME: Adding 1 to the atom type here, to differentiate between atom 0 and padding
                g.x = g.x + 1
                # edge_index = utils.add_self_loops(batch[i].edge_index, None, num_nodes =  max_len)[0]
                g = dense_transform(g)
                padded_x[i] = g.x

                # adj = utils.to_dense_adj(edge_index).squeeze()
                # FIXME: Adding 1 to the edge type here, to differentiate between padding and non-neighbors
                padded_adj[i, :num_nodes, :num_nodes] = g.adj[:num_nodes, :num_nodes] + 1
                # FIXME: Creating new edge type for ring connections (hard coded for 4 edge types)
                if self._add_rings == True:
                    padded_adj[i, :num_nodes, :num_nodes] += 4 * g.ring_adj
                # Adding a special edge type (2) for the diagonal
                padded_adj[i, :num_nodes, :num_nodes] += (g.adj[:num_nodes, :num_nodes] > 0)
                padded_adj[i, :num_nodes, :num_nodes].fill_diagonal_(2)


                mask[i] = g.mask
                if self.use_node_pe:
                    padded_p[i, :num_nodes] = g.node_pe
                if self.use_attention_pe:
                    attention_pe[i, :num_nodes, :num_nodes] = g.attention_pe
            return padded_x, padded_adj, padded_p, mask, attention_pe, default_collate(labels)
        return collate


def get_rings(edge_index, max_k=7):
    import graph_tool as gt
    import graph_tool.topology as top
    import networkx as nx

    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph_gt)
    gt.stats.remove_parallel_edges(graph_gt)
    # We represent rings with their original node ordering
    # so that we can easily read out the boundaries
    # The use of the `sorted_rings` set allows to discard
    # different isomorphisms which are however associated
    # to the same original ring – this happens due to the intrinsic
    # symmetries of cycles
    rings = set()
    sorted_rings = set()
    for k in range(3, max_k+1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                           generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))

    rings = list(rings)
    return rings