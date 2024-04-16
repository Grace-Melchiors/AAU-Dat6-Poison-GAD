import os
import yaml
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
from typing import Any, Dict, Sequence, Tuple, Union
from pygod.utils import load_data
import torch_geometric
from gad_adversarial_robustness.gad.dominant.means import ROBUST_MEANS, soft_weighted_medoid_k_neighborhood, chunked_message_and_aggregate
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, SparseTensor, Optional
from gad_adversarial_robustness.defenses.truncated_svd import get_truncated_svd
from gad_adversarial_robustness.defenses.jaccard import get_jaccard
from gad_adversarial_robustness.defenses.gdc import get_ppr_matrix
import math
from torch.nn import Parameter
from torch_geometric.nn import GCNConv


print(torch.cuda.is_available())

class RobustGraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super(RobustGraphConvolution, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))
        self._mean = soft_weighted_medoid_k_neighborhood
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        #support = torch.mm(input, self.weight)

        edge_weights = torch.ones((adj.size(1), ), dtype=torch.float).to("cuda")
        
        edge_index = SparseTensor.from_edge_index(adj, edge_weights, (input.size(0), input.size(0)))
        output = self._mean(edge_index, input)

        ##output = torch.spmm(adj, support)
        #if self.bias is not None:
        #    return output + self.bias
        #else:
        #    return output
        return output


class RGNNConv(GCNConv):
    """Extension of Pytorch Geometric's `GCNConv` to execute a robust aggregation function:
    - soft_k_medoid
    - soft_medoid (not scalable)
    - k_medoid
    - medoid (not scalable)
    - dimmedian

    Parameters
    ----------
    mean : str, optional
        The desired mean (see above for the options), by default 'soft_k_medoid'
    mean_kwargs : Dict[str, Any], optional
        Arguments for the mean, by default dict(k=64, temperature=1.0, with_weight_correction=True)
    """

    def __init__(self, in_channels: int, out_channels: int, mean='soft_k_medoid',
                 mean_kwargs: Dict[str, Any] = dict(k=64, temperature=1.0, with_weight_correction=True),
                 ):
        super().__init__(in_channels, out_channels, aggr='max')
        self._mean = soft_weighted_medoid_k_neighborhood
        #self._mean = soft_weighted_medoid_k_neighborhood
        self._mean_kwargs = mean_kwargs
        #self.do_chunk = True
        self.n_chunks = 8

    def message_and_aggregate(self, adj_t) -> torch.Tensor:
        return NotImplemented
    
    def propagate(self, edge_index: Union[torch.Tensor, SparseTensor], size=None, **kwargs) -> torch.Tensor:
        x = kwargs['x']
        #if not isinstance(edge_index, SparseTensor):
        #    edge_weights = kwargs['norm'] if 'norm' in kwargs else kwargs['edge_weight']
        #    A = SparseTensor.from_edge_index(edge_index, edge_weights, (x.size(0), x.size(0)))
        #    return self._mean(A, x, **self._mean_kwargs)
        edge_weights = kwargs['norm'] if 'norm' in kwargs else kwargs['edge_weight']
        edge_index = SparseTensor.from_edge_index(edge_index, edge_weights, (x.size(0), x.size(0)))

        def aggregate(edge_index: SparseTensor, x: torch.Tensor):
            return self._mean(edge_index, x, **self._mean_kwargs)
        #if self.do_chunk:
        #    return chunked_message_and_aggregate(edge_index, x, n_chunks=self.n_chunks, aggregation_function=aggregate)
        #print("returning aggr")
        return aggregate(edge_index, x)
        

    

class Encoder(nn.Module):
    def __init__(self, nfeat: int, nhid: int, dropout: float):
        super(Encoder, self).__init__()
        self.gc1 = RobustGraphConvolution(in_channels=nfeat, out_channels=nhid)
        self.gc2 = RobustGraphConvolution(in_channels=nhid, out_channels=nhid)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        gc1 = self.gc1(x, edge_index)
        #print(f'First element in embeddings after first forward pass: {gc1[0]}')
        x = F.relu(gc1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, edge_index))
        return x


class AttributeDecoder(nn.Module):
    def __init__(self, nfeat: int, nhid: int, dropout: float):
        super(AttributeDecoder, self).__init__()
        self.gc1 = GCNConv(in_channels=nhid, out_channels=nhid)
        self.gc2 = GCNConv(in_channels=nhid, out_channels=nfeat)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, edge_index))
        return x


class StructureDecoder(nn.Module):
    def __init__(self, nhid: int, dropout: float):
        super(StructureDecoder, self).__init__()
        self.gc1 = GCNConv(in_channels=nhid, out_channels=nhid)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T
        return x



"""
    svd_params : Dict[str, float], optional
        Parameters for the SVD preprocessing (`rank`), by default None
    jaccard_params : Dict[str, float], optional
        Parameters for the Jaccard preprocessing (`threshold`), by default None
"""

class Dominant(nn.Module):
    def __init__(self, feat_size: int, hidden_size: int, dropout: float, device: str, 
                 edge_index: torch.Tensor, adj_label: torch.Tensor, attrs: torch.Tensor, label: torch.Tensor, svd_params: Optional[Dict[str, float]] = None, jaccard_params: Optional[Dict[str, float]] = None, gdc_params: Optional[Dict[str, float]] = None):
        super(Dominant, self).__init__()
        self.device = device
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = AttributeDecoder(feat_size, hidden_size, dropout)
        self.struct_decoder = StructureDecoder(hidden_size, dropout)

        self.edge_index = edge_index.to(self.device)
        self.adj_label = adj_label.to(self.device).requires_grad_(True)
        self.attrs = attrs.to(self.device).requires_grad_(True)
        self.label = label.to(self.device)
        self._svd_params = svd_params
        self._jaccard_params = jaccard_params
        self._gdc_params = gdc_params

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Perform preprocessing such as SVD, GDC or Jaccard
        edge_index, edge_weight = self._preprocess_adjacency_matrix(edge_index, x)
        
        # Enforce that the input is contiguous
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)


        x = self.shared_encoder(x, edge_index)
        x_hat = self.attr_decoder(x, edge_index)
        struct_reconstructed = self.struct_decoder(x, edge_index)

        
        return struct_reconstructed, x_hat

    def fit(self, config: dict, verbose: bool = False):
        optimizer = torch.optim.Adam(self.parameters(), lr=config['model']['lr'])

        for epoch in range(config['model']['epochs']):
            self.train()
            optimizer.zero_grad()
            A_hat, X_hat = self.forward(self.attrs, self.edge_index)
            loss, struct_loss, feat_loss = loss_func(self.adj_label, A_hat, self.attrs, X_hat, config['model']['alpha'])
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"Epoch: {epoch:04d}, train_loss={loss.item():.5f}, "
                    f"train/struct_loss={struct_loss.item():.5f}, train/feat_loss={feat_loss.item():.5f}")

            if (epoch % 10 == 0 and verbose) or epoch == config['model']['epochs'] - 1:
                self.eval()
                A_hat, X_hat = self.forward(self.attrs, self.edge_index)
                loss, struct_loss, feat_loss = loss_func(self.adj_label, A_hat, self.attrs, X_hat, config['model']['alpha'])
                score = loss.detach().cpu().numpy()
                print(f"Epoch: {epoch:04d}, Auc: {roc_auc_score(self.label.detach().cpu().numpy(), score)}")

    def _ensure_contiguousness(self,
                               x: torch.Tensor,
                               edge_idx: torch.Tensor,
                               edge_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not x.is_sparse:
            x = x.contiguous()
        edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight

    def _preprocess_adjacency_matrix(self,
                                     edge_idx: torch.Tensor,
                                     x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_weight = None

        if self._svd_params is not None:
            adj = get_truncated_svd(
                torch.sparse.FloatTensor(
                    edge_idx,
                    torch.ones_like(edge_idx[0], dtype=torch.float32)
                ),
                **self._svd_params
            )
            for layer in self.layers:
                # the `get_truncated_svd` is incompatible with PyTorch Geometric due to negative row sums
                layer[0].normalize = False
            edge_idx, edge_weight = adj.indices(), adj.values()
            del adj
        elif self._jaccard_params is not None:
            #print(f'Jaccard w/ {self._jaccard_params["threshold"]}')
            adj = get_jaccard(
                torch.sparse.FloatTensor(
                    edge_idx,
                    torch.ones_like(edge_idx[0], dtype=torch.float32)
                ),
                x,
                **self._jaccard_params
            ).coalesce()
            edge_idx, edge_weight = adj.indices(), adj.values()
            del adj
        
        elif self._gdc_params is not None:
            adj = get_ppr_matrix(
                torch.sparse.FloatTensor(edge_idx, torch.ones_like(edge_idx[0], dtype=torch.float32)),
                **self._gdc_params,
                normalize_adjacency_matrix=True
            )
            edge_idx, edge_weight = adj.indices(), adj.values()
            del adj


        return edge_idx, edge_weight


def normalize_adj(adj: np.ndarray) -> sp.coo_matrix:
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def loss_func(adj: torch.Tensor, A_hat: torch.Tensor, attrs: torch.Tensor, X_hat: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors
    return cost, structure_cost, attribute_cost


def load_anomaly_detection_dataset(dataset: Data, datadir: str = 'data', device: Optional[str] = 'cuda') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge_index = dataset.edge_index
    adj = to_dense_adj(edge_index)[0].cpu().numpy()

    feat = dataset.x.detach().cpu().numpy()
    truth = dataset.y.bool().detach().cpu().numpy().flatten()

    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0])).toarray()
    adj = adj + np.eye(adj.shape[0])
    return adj_norm, feat, truth, adj

if __name__ == '__main__':
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the YAML file relative to the script's directory
    yaml_path = os.path.join(script_dir, '..', '..', '..', 'configs', 'dominant_config.yaml')
    with open(yaml_path) as file:
        config = yaml.safe_load(file)

    dataset: Data = load_data("inj_cora")
    adj, _, _, adj_label = load_anomaly_detection_dataset(dataset, config['model']['device'])
    #edge_index = torch.LongTensor(np.array(sp.coo_matrix(adj).nonzero()))
    adj_label = torch.FloatTensor(adj_label).to(config['model']['device'])
    #attrs = torch.FloatTensor(attrs)

    edge_index = dataset.edge_index.to(config['model']['device'])
    label = torch.Tensor(dataset.y.bool()).to(config['model']['device'])
    attrs = dataset.x.to(config['model']['device'])

    jaccard_params = {
        "threshold": 0.01
    }

    gdc_params = {
        "alpha": 0.15,
        "k": 64
    }

    model = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                     device=config['model']['device'], edge_index=edge_index, adj_label=adj_label, attrs=attrs, label=label)
    model.to(config['model']['device'])
    model.fit(config, verbose=True)

    self.weights