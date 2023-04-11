import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import to_dense_batch

from graphgps.layer.gps_layer import GPSLayer
# from graphgps.utils import CudaTimer


def add_graph_token(data, token):
    """Helper function to augment a batch of PyG graphs
    with a graph token each. Note that the token is
    automatically replicated to fit the batch.
    Args:
        data: A PyG data object holding a single graph
        token: A tensor containing the graph token values
    Returns:
        The augmented data object.
    """
    B = len(data.batch.unique())
    tokens = torch.repeat_interleave(token, B, 0)
    data.x = torch.cat([tokens, data.x], 0)
    data.batch = torch.cat(
        [torch.arange(0, B, device=data.x.device, dtype=torch.long), data.batch]
    )
    data.batch, sort_idx = torch.sort(data.batch)
    data.x = data.x[sort_idx]
    return data

def add_virtual_edges(dense_edges, token):
    n_batch, num_nodes, _, out_dim = dense_edges.shape
    token = token.expand(n_batch, out_dim)
    dense_edges = torch.cat([dense_edges, token.unsqueeze(1).expand(n_batch, num_nodes, out_dim).unsqueeze(1)], dim=1)
    dense_edges = torch.cat([dense_edges, token.unsqueeze(1).expand(n_batch, num_nodes+1, out_dim).unsqueeze(2)], dim=2)
    return dense_edges

class CLSEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, embed_dim):
        super(CLSEncoder, self).__init__()
        self.node_token = torch.nn.Parameter(torch.zeros(1, embed_dim))
        self.edge_att_token = torch.nn.Parameter(torch.zeros(1, embed_dim))
        self.edge_value_token = torch.nn.Parameter(torch.zeros(1, embed_dim))

    def forward(self, batch):
        batch = add_graph_token(batch, self.node_token)
        batch.edge_attention = add_virtual_edges(batch.edge_attention, self.edge_att_token)
        batch.edge_values = add_virtual_edges(batch.edge_values, self.edge_value_token)
        return batch

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.posenc_RWSE.enable and not cfg.posenc_RWSE.precompute:
            self.rwse_compute = register.edge_encoder_dict['RWSEonthefly']()
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-set edge dim for PNA.
            cfg.gnn.dim_edge = 16 if 'PNA' in cfg.gt.layer_type else cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
        if cfg.model.graph_pooling == 'first':
            self.cls_encoder = CLSEncoder(cfg.gnn.dim_inner)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        h_dense, mask = to_dense_batch(batch.x, batch.batch)
        num_nodes = mask.size()[1]
        batch.attn_mask = mask.view(-1, num_nodes, 1) * mask.view(-1, 1, num_nodes)
        return batch


@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(GPSLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                act=cfg.gnn.act,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                graphiT_share=cfg.dataset.edge_encoder_shared
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            # with CudaTimer(type(module).__name__) as _:
            batch = module(batch)
        # import pdb; pdb.set_trace()
        return batch
