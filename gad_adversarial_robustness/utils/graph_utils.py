from torch_sparse import SparseTensor 
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

import math
import torch
from typing import Union
import numpy as np
import torch_sparse
from torch_geometric.data.dataset import Dataset, BaseData
import scipy.sparse as sp
from torch_geometric.utils import to_dense_adj


from torch_geometric.datasets import AttributedGraphDataset

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

# TODO: Probably move to vis_utils or something like that.
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

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def load_anomaly_detection_dataset(dataset, datadir='data'):
    # import dataset and extract its parts
    #dataset = load_data("inj_cora")
    edge_index = dataset.edge_index
    adj = to_dense_adj(edge_index)[0].detach().cpu().numpy()

    
    feat= dataset.x.detach().cpu().numpy()
    # remember to use .bool() if the dataset is an injected dataset, to enable binary labels.
    # TODO: handle the case where we inject ourselves
    truth = dataset.y.bool().detach().cpu().numpy()
    truth = truth.flatten()

    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    adj = adj + np.eye(adj.shape[0])
    return adj_norm, feat, truth, adj


def get_n_anomaly_indexes(truth, n_anomalies):
    indexes = np.where(truth == 1)[0]
    # get the first n_anomalies indexes
    if n_anomalies != 999:

        indexes = indexes[:n_anomalies]

    print(f'Anomalies indexes: {indexes}')
    return indexes


def top_anomalous_nodes(anomaly_scores, labels, K, print_scores = False):
    # Get indices of nodes with label = 1
    anomaly_indices = np.where(labels == 1)[0]

    # Sort anomaly scores in descending order and get the corresponding indices
    sorted_indices = np.argsort(anomaly_scores)[::-1]

    # Filter the sorted indices to keep only those with label = 1
    filtered_indices = np.intersect1d(sorted_indices, anomaly_indices)

    # Get the top K indices
    top_K_indices = filtered_indices[:K]

    if print_scores:
        # Print anomaly scores for the top K indices
        print("Anomaly scores for top K indices with label = 1:")
        for idx in top_K_indices:
            print("Index:", idx, "| Anomaly Score:", anomaly_scores[idx])

    return top_K_indices



def get_anomaly_indexes(scores, labels, k, method='top', print_scores=False, random_top = False):
    # Convert lists to numpy arrays
    scores = np.array(scores)
    labels = np.array(labels)
    np.random.seed(123)
    
    # Find indices of nodes with label = 1 and non-nan scores
    valid_indices = np.where((labels == 1) & (~np.isnan(scores)))[0]
    threshold = np.percentile(scores[valid_indices], 75)

    if method == 'top':
        if random_top:
            # Randomly sample k anomalies from the top 75% of scores where label == 1
            top_indices = valid_indices[scores[valid_indices] >= threshold]
            selected_indices = np.random.choice(top_indices, k, replace=False)
        else:
            # Sort the valid indices based on scores in descending order, ignoring NaN values
            sorted_indices = sorted(valid_indices, key=lambda i: scores[i] if not np.isnan(scores[i]) else -np.inf, reverse=True)
            # Extract K indices
            selected_indices = sorted_indices[:k]

    elif method == 'lowest':
        # Sort the valid indices based on scores in ascending order, ignoring NaN values
        sorted_indices = sorted(valid_indices, key=lambda i: scores[i] if not np.isnan(scores[i]) else np.inf)
    elif method == 'normal':
        # Sample K indices from a normal distribution centered around the mean of scores
        mean_score = np.mean(scores[valid_indices])
        std_dev = np.std(scores[valid_indices])
        sampled_indices = np.random.normal(mean_score, std_dev, k)
        # Clip the sampled indices to be within the valid range
        clipped_indices = np.clip(sampled_indices, 0, len(valid_indices)-1)
        # Round and convert to integers
        sorted_indices = np.argsort(np.abs(clipped_indices - mean_score))[:k]
    
        # Extract K indices
        selected_indices = sorted_indices[:k]

    if print_scores:
        # Print anomaly scores for the top K indices
        print("Anomaly scores for top K indices with label = 1:")
        for idx in selected_indices:
            print("Index:", idx, "| Anomaly Score:", scores[idx])
    
    return selected_indices



def get_anomalies_with_label_1(scores, labels):
    # Create a list of tuples (index, label, score)
    nodes = [(i, labels[i], scores[i]) for i in range(len(scores))]
    
    # Filter nodes with label = 1
    nodes_with_label_1 = filter(lambda x: x[1] == 1, nodes)
    
    # Sort the filtered list based on anomaly scores in descending order
    sorted_nodes = sorted(nodes_with_label_1, key=lambda x: x[2], reverse=True)
    
    # Print all anomaly scores that have label = 1
    for node in sorted_nodes:
        print("Node index: {}, Score: {}".format(node[0], node[2]))


def load_injected_dataset(dataset_name, seed = 1234):

    if dataset_name == 'Wiki':
        #data = AttributedGraphDataset(root = "data/"+dataset_name, name = dataset_name)
        data = AttributedGraphDataset('data/', 'Wiki', transform=T.NormalizeFeatures())[0]
    if dataset_name == 'Cora':
        #data = Planetoid(root='data/', name=dataset_name)
        data = Planetoid('data/', 'cora', transform=T.NormalizeFeatures())[0]
    if dataset_name == 'Facebook':
        data = Planetoid('data/', 'CiteSeer', transform=T.NormalizeFeatures())[0]
        print("NUM NODES CITE SEER: ", data.num_nodes)
    if dataset_name == 'Amazon':
        from torch_geometric.datasets import Amazon
        from torch_geometric.utils import subgraph
        dataset = Amazon(root='data/Amazon', name='Computers')
        data = dataset[0]
        num_nodes_to_sample = 2000
        torch.manual_seed(123)
        node_indices = torch.randperm(data.num_nodes)[:num_nodes_to_sample]
        data.edge_index = subgraph(node_indices, data.edge_index)[0]





    num_nodes_to_inject = math.ceil((data.num_nodes / 100) * 5)
    num_nodes_per_clique = 10
    num_cliques = (num_nodes_to_inject // 2) // num_nodes_per_clique
    # Temp
    num_cliques = 7
    num_contextual_outliers = num_nodes_to_inject - num_cliques * num_nodes_per_clique
    print("num contextual outliers: ", num_contextual_outliers)
    print("num structural outliers: ", num_cliques*num_nodes_per_clique)

    prior_labels = data.y

    data, ya = gen_contextual_outlier(data, n=num_contextual_outliers, k=10, seed=12345)
    #n (int) – Number of nodes converting to outliers.
    #k (int) – Number of candidate nodes for each outlier node.

    data, ys = gen_structural_outlier(data, m=num_nodes_per_clique, n=num_cliques, p=0.2, seed=1234)
    #m (int) - Number nodes in the outlier cliques.
    #n (int) - Number of outlier clique

    data.y = torch.logical_or(ys, ya).long()
    # Count occurrences of value 1
    data.y = data.y.bool()
    count_ones = (data.y == 1).sum().item()

    print("Occurrences of value 1 hi:", count_ones)
        
    return data, prior_labels