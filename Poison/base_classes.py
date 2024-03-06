from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List, Dict
from torchtyping import TensorType, patch_typeguard
from torch_sparse import SparseTensor
import scipy.sparse as sp
import torch

class Poison(ABC):
    def __init__(
        self,
        adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
        attr: TensorType["n_nodes", "n_features"],
        labels: TensorType["n_nodes"],
        **kwargs
    ):
        self.adj = adj
        self.attr = attr
        self.labels = labels


class LocalPoison(Poison, ABC):
    def __init__(
        self,
        adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"], sp.csr_matrix],
        target_node_id,
        **kwargs
    ):
        self.target_node_id = target_node_id

        super().__init__(adj, **kwargs)

        edge_index_rows, edge_index_cols, edge_weight = adj.coo()
        self.edge_index = torch.stack([edge_index_rows, edge_index_cols], dim=0).to(
            "cpu"
        )
        self.edge_weight = edge_weight.to("cpu")
        self.n = adj.size(0)
        self.d = self.attr.shape[1]
