from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List, Dict
from torchtyping import TensorType, patch_typeguard
from torch_sparse import SparseTensor
import scipy.sparse as sp
import torch
import copy
from models import MODEL_TYPE
from pygod.detector import DOMINANT
from pygod.utils import load_data
from torch.utils.data import Dataset


import numpy as np
import torch_sparse

class BasePoison(ABC):  #ABC means abstract base class
    def __init__(self, data, budget = 0, target_node_indexes = []):
        self.data = data
        self.budget = budget
        self.target_node_indexes = target_node_indexes

        # Get individual properties
        edge_index = data.edge_index.cpu()

        if hasattr(data, "__num_nodes__"):
            num_nodes = data.__num_nodes__
        else:
            num_nodes = data.num_nodes

        #if dataset.edge_attr is None:
            # If we have no edge attributes (in datasets such as Cora)
        edge_weight = torch.ones(edge_index.size(1))
        #else:
            #edge_weight = dataset.edge_attr
        edge_weight = edge_weight.cpu()

        # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
        adj = sp.csr_matrix((edge_weight, edge_index), (num_nodes, num_nodes))
        adj.data = np.ones_like(adj.data)
        adj = torch_sparse.SparseTensor.from_scipy(adj).coalesce().to("cpu")

        # Extract attributes
        node_attr_matrix = data.x.cpu().numpy()
        self.node_attr = torch.from_numpy(node_attr_matrix).to("cpu")
        self.labels = data.y.squeeze().to("cpu")
        self.adj = adj
    @abstractmethod
    def poison_data(self):
        pass


class Poison(ABC):
    def __init__(
        self,
        adj: SparseTensor,
        #Union[SparseTensor, TensorType["n_nodes", "n_nodes"], sp.csr_matrix],
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
        adj: SparseTensor,
        #Union[SparseTensor, TensorType["n_nodes", "n_nodes"], sp.csr_matrix],
        attr: TensorType["n_nodes", "n_features"],
        labels: TensorType["n_nodes"],
        target_node_id,
        attacked_model: Optional[MODEL_TYPE] = None,
        data: Optional[Dataset] = None
    ):

        super().__init__(adj, attr, labels)

        self.target_node_id = target_node_id
        edge_index_rows, edge_index_cols, edge_weight = adj.coo()
        self.edge_index = torch.stack([edge_index_rows, edge_index_cols], dim=0).to(
            "cpu"
        )
        self.edge_weight = edge_weight.to("cpu")
        self.n = adj.size(0)
        self.d = self.attr.shape[1]
        if attacked_model is None:
            self.attacked_model = copy.deepcopy("cpu").to(self.device)
        else:
            # The case where we have a modified surrogate model. I.e. DOMINANT w/ 2 layers
            self.attacked_model = attacked_model

        if data is not None:
            self.data = data
        
        #self.attacked_model.eval()
        #for p in self.attacked_model.parameters():
        #    p.requires_grad = False
        self.eval_model = self.attacked_model

        if type(attacked_model) is DOMINANT:
            self.attacked_model.decision_function(data)


        @abstractmethod
        def poison(self, n_perturbations: int, node_idx: int, **kwargs):
            pass
