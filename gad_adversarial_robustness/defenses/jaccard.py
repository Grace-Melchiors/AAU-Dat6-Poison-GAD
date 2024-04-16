import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import from_scipy_sparse_matrix, add_remaining_self_loops
import torch_scatter
import torch_sparse


def get_jaccard(adjacency_matrix: torch.Tensor, features: torch.Tensor, threshold: int = 0.01):
    """Jaccard similarity edge filtering as proposed in Huijun Wu, Chen Wang, Yuriy Tyshetskiy, Andrew Docherty, Kai Lu,
    and Liming Zhu.  Adversarial examples for graph data: Deep insights into attack and defense.

    Parameters
    ----------
    adjacency_matrix : torch.Tensor
        Sparse [n,n] adjacency matrix.
    features : torch.Tensor
        Dense [n,d] feature matrix.
    threshold : int, optional
        Similarity threshold for filtering, by default 0.

    Returns
    -------
    torch.Tensor
        Preprocessed adjacency matrix.
    """
    #print("IN JACCAD")
    row, col = adjacency_matrix._indices().cpu()
    values = adjacency_matrix._values().cpu()
    N = adjacency_matrix.shape[0]

    if features.is_sparse:
        features = features.to_dense()

    modified_adj = sp.coo_matrix((values.numpy(), (row.numpy(), col.numpy())), (N, N))
    modified_adj = drop_dissimilar_edges(features.detach().cpu().numpy(), modified_adj, threshold=threshold)
    modified_adj = torch.sparse.FloatTensor(*from_scipy_sparse_matrix(modified_adj)).to(adjacency_matrix.device)
    return modified_adj




def drop_dissimilar_edges(features, adj, threshold: int = 0):
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    modified_adj = adj.copy().tolil()

    edges = np.array(modified_adj.nonzero()).T
    removed_cnt = 0
    features = sp.csr_matrix(features)
    for edge in edges:
        n1 = edge[0]
        n2 = edge[1]
        if n1 > n2:
            continue

        J = _jaccard_similarity(features[n1], features[n2])

        if J <= threshold:
            modified_adj[n1, n2] = 0
            modified_adj[n2, n1] = 0
            removed_cnt += 1
    return modified_adj


def _jaccard_similarity(a, b):
    intersection = a.multiply(b).count_nonzero()
    J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
    return J
