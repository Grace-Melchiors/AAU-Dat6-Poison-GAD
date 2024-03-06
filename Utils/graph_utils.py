from torch_sparse import SparseTensor 
import torch
from typing import Union
import numpy as np
import torch_sparse
from torch_geometric.data.dataset import Dataset, BaseData

def adj_matrix_sparse_coo_to_dense(sparse_tensor: SparseTensor):
    # Unsqueeze to treat each tensor as a separate matrix, such that each matrix is stacked along the first dimension
    col = sparse_tensor.storage.col().unsqueeze(0)
    row = sparse_tensor.storage.row().unsqueeze(0)
    concatenated_tensor = torch.cat((col, row), dim=0)

    return concatenated_tensor

def prepare_graph(dataset: Union[Dataset, BaseData]):
    edge_index = dataset.edge_index.cpu()

    if hasattr(dataset, "__num_nodes__"):
        num_nodes = dataset.__num_nodes__
    else:
        num_nodes = dataset.num_nodes

    if dataset.edge_attr is None:
        # If we have no edge attributes (in datasets such as Cora)
        edge_weight = torch.ones(edge_index.size(1))
    else:
        edge_weight = dataset.edge_attr
    edge_weight = edge_weight.cpu()

    # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    adj = sp.csr_matrix((edge_weight, edge_index), (num_nodes, num_nodes))
    adj.data = np.ones_like(adj.data)
    adj = torch_sparse.SparseTensor.from_scipy(adj).coalesce().to("cpu")

    # Extract attributes
    node_attr_matrix = dataset.x.cpu().numpy()
    node_attr = torch.from_numpy(node_attr_matrix).to("cpu")
    labels = dataset.y.squeeze().to("cpu")
    print("node_attrs")
    print(node_attr_matrix)
    print("adj")
    print(adj)
    print("labels")
    print(labels)

    return node_attr, adj, labels
    

