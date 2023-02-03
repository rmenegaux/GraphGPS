import logging

import torch
from torch_geometric.utils import subgraph, coalesce
from tqdm import tqdm


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data

def add_rings(data, config):
    # Find all pairs of nodes that are part of the same ring
    rings = get_rings(data.edge_index, max_k=config.rings_max_length)
    ring_connections = set()
    for ring in rings:
        for i in ring:
            for j in ring:
                if i < j:
                    ring_connections.add((i, j))
                    ring_connections.add((j, i))
    # Store these indices in a new ring_index graph attribute
    data.ring_index = data.edge_index.new_zeros(2, len(ring_connections))
    zipped_connections = list(zip(*ring_connections)) if len(ring_connections) > 0 else [(), ()]
    data.ring_index[0], data.ring_index[1] = torch.Tensor(zipped_connections[0]), torch.Tensor(zipped_connections[1])
    # Create new edges for each ring connection
    # aka merge ring_index into edge_index
    if config.rings_coalesce_edges == True:
        data.edge_index = torch.cat([data.edge_index, data.ring_index], dim=1)
        # FIXME: This should be done on a per-dataset basis. Ex if edge_attr is multi-dimensional...
        # Shady stuff right here
        # In the case of multi-dimensional edge_attr, could add a column of 0s to existing edge_attr,
        # and set ring_attr to (0, ..., 0, 1)
        ring_attr = data.edge_attr.new_zeros(len(ring_connections))
        ring_attr += config.edge_encoder_num_types
        data.edge_attr = torch.cat([data.edge_attr, ring_attr], dim=0)
        # Merge duplicate entries, by summing their edge_attr.
        data.edge_index, data.edge_attr = coalesce(data.edge_index, data.edge_attr)
    return data

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

def get_SPD(edge_index):
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
    # Ok so now we have the graph entity to work on with networkx
    import pdb;pdb.set_trace()
    nodes = set(edge_index[:,0]) # all nodes
    SPD = nx.all_shortest_paths_lengths(graph_gt, nodes, nodes)
    return SPD

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
