from torch_geometric.graphgym.register import register_pooling
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch


@register_pooling('example')
def global_example_pool(x, batch, size=None):
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')

@register_pooling('first')
def graph_token_pooling(x, batch, *args):
    """Extracts the graph token from a batch to perform graph-level prediction.
    Typically used together with Graphormer when GraphormerEncoder is used and
    the global graph token is used: `cfg.graphormer.use_graph_token == True`.
    """
    x, _ = to_dense_batch(x, batch)
    return x[:, 0, :]
