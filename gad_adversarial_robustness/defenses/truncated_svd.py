import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import from_scipy_sparse_matrix, add_remaining_self_loops
import torch_scatter
import torch_sparse


def get_truncated_svd(adjacency_matrix: torch.Tensor, rank: int = 50):
    """Truncated SVD preprocessing as proposed in Negin Entezari, Saba A. Al - Sayouri, Amirali Darvishzadeh, and
    Evangelos E. Papalexakis. All you need is Low(rank):  Defending against adversarial attacks on graphs.

    Attention: the result will not be sparse!

    Parameters
    ----------
    adjacency_matrix : torch.Tensor
        Sparse [n,n] adjacency matrix.
    rank : int, optional
        Rank of the truncated SVD, by default 50.

    Returns
    -------
    torch.Tensor
        Preprocessed adjacency matrix.
    """
    row, col = adjacency_matrix._indices().cpu()
    values = adjacency_matrix._values().cpu()
    N = adjacency_matrix.shape[0]

    low_rank_adj = sp.coo_matrix((values.numpy(), (row.numpy(), col.numpy())), (N, N))
    low_rank_adj = truncatedSVD(low_rank_adj, rank)
    low_rank_adj = torch.from_numpy(low_rank_adj).to(adjacency_matrix.device, adjacency_matrix.dtype)

    return svd_norm_adj(low_rank_adj).to_sparse()


def svd_norm_adj(adj: torch.Tensor):
    mx = adj + torch.eye(adj.shape[0]).to(adj.device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx


def truncatedSVD(data, k=50):
    if sp.issparse(data):
        data = data.asfptype()
        U, S, V = sp.linalg.svds(data, k=k)
        diag_S = np.diag(S)
    else:
        U, S, V = np.linalg.svd(data)
        U = U[:, :k]
        S = S[:k]
        V = V[:k, :]
        diag_S = np.diag(S)

    return U @ diag_S @ V