from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List, Dict
from torchtyping import TensorType, patch_typeguard
from torch_sparse import SparseTensor
import scipy.sparse as sp
import torch

class Poison(ABC):
    def __init__(
        self,
        adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"], sp.csr_matrix],
        attr: TensorType["n_nodes", "n_features"],
        labels: TensorType["n_nodes"],
        nnodes: Optional[int] = None,
        attack_structure: bool = True,
        attack_features: bool = False

    ):
        self.adj = adj
        self.attr = attr
        self.labels = labels
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features



class LocalPoison(Poison, ABC):
    def __init__(
        self,
        adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"], sp.csr_matrix],
        attr: TensorType["n_nodes", "n_features"],
        labels: TensorType["n_nodes"],
        target_node_id,
    ):
        self.target_node_id = target_node_id

        super().__init__(adj, attr, labels)

        edge_index_rows, edge_index_cols, edge_weight = adj.coo()
        self.edge_index = torch.stack([edge_index_rows, edge_index_cols], dim=0).to(
            "cpu"
        )
        self.edge_weight = edge_weight.to("cpu")
        self.n = adj.size(0)
        self.d = self.attr.shape[1]

        @abstractmethod
        def poison(self, n_perturbations: int, node_idx: int, **kwargs):
            pass
