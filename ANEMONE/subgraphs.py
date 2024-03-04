import torch
from torch import Tensor

from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils.map import map_index
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes




def get_node_neighboors(node, edge_index):
    """ node (int): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        
        returns (LongTensor): the list of nodes connected """
    
    if subset.dtype != torch.bool:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = index_to_mask(subset, size=num_nodes)
    else:
        num_nodes = subset.size(0)
        node_mask = subset
        subset = node_mask.nonzero().view(-1)

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]

    print(edge_index)
    print(edge_index[1])

    return edge_index[1]


edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                           [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
subset = torch.tensor([3, 4, 5])


get_node_neighboors(3, edge_index)