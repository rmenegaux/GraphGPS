import torch

from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('RingEdge')
class RingEdgeEncoder(torch.nn.Module):
    '''
    Transforms dense edge features by adding a vector to `batch.edge_dense`
    for every pair of nodes (i, j) that are in a ring

    Supposes `batch.ring_index` has been set.
    These are precomputed if the following config option is set:
        cfg.dataset.rings = True

    Current version requires that dense edge features have been computed 
    (`batch.edge_dense` set via the DenseEdgeEncoder or the RPE modules)
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