import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import from_scipy_sparse_matrix, add_remaining_self_loops
import torch_scatter
import torch_sparse


def get_ppr_matrix(adjacency_matrix: torch.Tensor,
                   alpha: float = 0.15,
                   k: int = 32,
                   normalize_adjacency_matrix: bool = False,
                   use_cpu: bool = False,
                   **kwargs) -> torch.Tensor:
    """Calculates the personalized page rank diffusion of the adjacency matrix as proposed in Johannes Klicpera,
    Stefan Weißenberger, and Stephan Günnemann. Diffusion Improves Graph Learning.

    Parameters
    ----------
    adjacency_matrix : torch.Tensor
        Sparse adjacency matrix.
    alpha : float, optional
        Teleport probability, by default 0.15.
    k : int, optional
        Neighborhood for sparsification, by default 32.
    normalize_adjacency_matrix : bool, optional
        Should be true if the adjacency matrix is not normalized via two-sided degree normalization, by default False.
    use_cpu : bool, optional
        If True the matrix inverion will be performed on the CPU, by default False.

    Returns
    -------
    torch.Tensor
        Preprocessed adjacency matrix.
    """
    dim = -1

    assert alpha > 0 and alpha < 1
    assert k >= 1
    if use_cpu:
        device = adjacency_matrix.device
        adjacency_matrix = adjacency_matrix.cpu()

    dtype = adjacency_matrix.dtype

    if normalize_adjacency_matrix:
        if adjacency_matrix.is_sparse:
            adjacency_matrix = adjacency_matrix.to_dense()
        adjacency_matrix += torch.eye(*adjacency_matrix.shape, device=adjacency_matrix.device, dtype=dtype)
        D_tilde = torch.diag(1 / torch.sqrt(adjacency_matrix.sum(axis=1)))
        adjacency_matrix = D_tilde @ adjacency_matrix @ D_tilde
        del D_tilde

    adjacency_matrix = alpha * torch.inverse(
        torch.eye(*adjacency_matrix.shape, device=adjacency_matrix.device, dtype=dtype)
        - (1 - alpha) * adjacency_matrix
    )

    if use_cpu:
        adjacency_matrix = adjacency_matrix.to(device)

    selected_vals, selected_idx = torch.topk(adjacency_matrix, int(k), dim=dim)
    norm = selected_vals.sum(dim)
    norm[norm <= 0] = 1
    selected_vals /= norm[:, None]

    row_idx = torch.arange(adjacency_matrix.size(0), device=adjacency_matrix.device)[:, None]\
        .expand(adjacency_matrix.size(0), int(k))
    return torch.sparse.FloatTensor(
        torch.stack((row_idx.flatten(), selected_idx.flatten())),
        selected_vals.flatten()
    ).coalesce()