import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import pygod


import torch_geometric
from torch_geometric.datasets import Planetoid
from pygod.utils import load_data
from torch_geometric.utils import to_dense_adj

def load_anomaly_detection_dataset(dataset='Cora', datadir='data'):
    # dataset = Planetoid(root=datadir, name=dataset)
    # data = dataset[0]
    data = load_data('inj_cora')

    edge_index = data.edge_index   # adjacency matrix

    # .detach().cpu().numpy() --> converts the tensor to a numpy array
    adj = to_dense_adj(edge_index)[0].detach().cpu().numpy()

    adj_norm = normalize_adj(adj) # 
    adj_norm = adj_norm.toarray()

    # again detach from tensor to numpy array
    feat = data.x.detach().cpu().numpy()   # node features
    truth = data.y.bool()  # truth labels

    return adj_norm, feat, truth, adj


# def load_dataset(dataset, datadir='AnomalyDAE\\data\\Cora'):
#     """
#     import dataset....
#     """
#     TODO: "implement"


def normalize_adj(adj):
    """
    symetrically normalizes an adjacency matrix

    Symmetric normalization of the adjacency matrix is a 
    common preprocessing step in graph neural networks to 
    ensure stable training and better representation learning.

    by normalizing, the model can effectively capture the graph structure and relationships between nodes
    """
    # converts input matrix to sparse matrix
    adj = sp.coo_matrix(adj)

    # Calculate the sum of each row in the adjacency matrix to obtain rowsum
    rowsum = np.array(adj.sum(1))

    # Computes the inverse square root of the row sums and handles cases where the inverse sqrt is infinite by setting to 0
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0

    # a diagonal matrix with the inverse sqrt values
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # returns symmestrically normalized adjacency matrix in COO format
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()