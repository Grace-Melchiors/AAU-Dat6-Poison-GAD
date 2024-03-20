from torch_sparse import SparseTensor 
import torch
from typing import Union
import numpy as np
import torch_sparse
from torch_geometric.data.dataset import Dataset, BaseData
import scipy.sparse as sp

# For drawing graph, maybe move to seperate file for display/graph utils
import networkx as nx
import matplotlib.pyplot as plt

from pygod.generator import gen_contextual_outlier, gen_structural_outlier


def insert_anomalies(data, ctx_n=None, ctx_k=None, struct_m=None, struct_n=None):
    if ctx_n and ctx_k and struct_m and struct_n is not None:
        data, ya = gen_contextual_outlier(data, n=ctx_n, k=ctx_k)
        data, ys = gen_structural_outlier(data, m=struct_m, n=struct_n)
        y = torch.logical_or(ys, ya).long()
        data.y = y
    elif ctx_n and ctx_k is not None:
        data, ya = gen_contextual_outlier(data, n=ctx_n, k=ctx_k)
        data.y = ya
    elif struct_m and struct_n is not None:
        data, ys = gen_structural_outlier(data, m=struct_m, n=struct_n)
        data.y = ys
    else:
        raise ValueError("Wrong combination of input parameters")

    
    return data


def adj_matrix_sparse_coo_to_dense(sparse_tensor: SparseTensor):
    # Unsqueeze to treat each tensor as a separate matrix, such that each matrix is stacked along the first dimension
    col = sparse_tensor.storage.col().unsqueeze(0)
    row = sparse_tensor.storage.row().unsqueeze(0)
    concatenated_tensor = torch.cat((col, row), dim=0)

    return concatenated_tensor

def convert_scipy_sparse_matrix_to_sparse_tensor(spmat: sp.spmatrix, grad: bool = False):
    """

    Convert a scipy.sparse matrix to a SparseTensor.
    Parameters
    ----------
    spmat: sp.spmatrix
        The input (sparse) matrix.
    grad: bool
        Whether the resulting tensor should have "requires_grad".
    Returns
    -------
    sparse_tensor: SparseTensor
        The output sparse tensor.
    """
    if str(spmat.dtype) == "float32":
        dtype = torch.float32
    elif str(spmat.dtype) == "float64":
        dtype = torch.float64
    elif str(spmat.dtype) == "int32":
        dtype = torch.int32
    elif str(spmat.dtype) == "int64":
        dtype = torch.int64
    elif str(spmat.dtype) == "bool":
        dtype = torch.uint8
    else:
        dtype = torch.float32
    return SparseTensor.from_scipy(spmat).to(dtype).coalesce()

def prepare_graph(dataset: Union[Dataset, BaseData]):
    edge_index = dataset.edge_index.cpu()

    if hasattr(dataset, "__num_nodes__"):
        num_nodes = dataset.__num_nodes__
    else:
        num_nodes = dataset.num_nodes

    #if dataset.edge_attr is None:
        # If we have no edge attributes (in datasets such as Cora)
    edge_weight = torch.ones(edge_index.size(1))
    #else:
        #edge_weight = dataset.edge_attr
    edge_weight = edge_weight.cpu()

    print(edge_weight)
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
    




def draw_dataset_graph(graph, graph_name=None):
    # Draws the graph ----
    plt.figure(figsize=(8, 8))
    nx.draw(graph, node_size=10, node_color='b', edge_color='gray', with_labels=False)
    if (graph_name is None) : 
        plt.title('Dataset Graph')
    else :
        plt.title('CORA Dataset Graph')
    plt.show()
