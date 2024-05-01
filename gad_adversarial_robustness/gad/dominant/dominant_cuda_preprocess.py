import os
import time
from pygod.detector import DOMINANT
import yaml
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_edge_index, add_self_loops
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from typing import Optional, Tuple
from pygod.utils import load_data
from torch.nn.modules import Module
from torch.nn import Parameter
import math
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torch_geometric.utils import from_scipy_sparse_matrix, add_remaining_self_loops

def drop_dissimilar_edges(features, adj, threshold: int = 0.02):
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

def get_jaccard(adjacency_matrix: torch.Tensor, features: torch.Tensor, threshold: int = 0.02):
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
    row, col = adjacency_matrix._indices().cpu()
    values = adjacency_matrix._values().cpu()
    N = adjacency_matrix.shape[0]

    if features.is_sparse:
        features = features.to_dense()

    modified_adj = sp.coo_matrix((values.numpy(), (row.numpy(), col.numpy())), (N, N))
    modified_adj = drop_dissimilar_edges(features.cpu().numpy(), modified_adj, threshold=threshold)
    modified_adj = torch.sparse.FloatTensor(*from_scipy_sparse_matrix(modified_adj)).to(adjacency_matrix.device)
    return modified_adj


class Encoder(nn.Module):
    def __init__(self, nfeat: int, nhid: int, dropout: float):
        super(Encoder, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight) -> torch.Tensor:
        x = F.relu(self.gc1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, edge_index, edge_weight=edge_weight))
        return x


class AttributeDecoder(nn.Module):
    def __init__(self, nfeat: int, nhid: int, dropout: float):
        super(AttributeDecoder, self).__init__()
        self.gc1 = GCNConv(nhid, nhid)
        self.gc2 = GCNConv(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight) -> torch.Tensor:
        x = F.relu(self.gc1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, edge_index, edge_weight=edge_weight))
        return x


class StructureDecoder(nn.Module):
    def __init__(self, nhid: int, dropout: float):
        super(StructureDecoder, self).__init__()
        self.gc1 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight) -> torch.Tensor:
        x = F.relu(self.gc1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T
        return x


class Dominant(nn.Module):
    def __init__(self, feat_size: int, hidden_size: int, dropout: float, device: str, 
                 edge_index: torch.Tensor, adj_label: torch.Tensor, attrs: torch.Tensor, label: np.ndarray):
        super(Dominant, self).__init__()
        self.device = device
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = AttributeDecoder(feat_size, hidden_size, dropout)
        self.struct_decoder = StructureDecoder(hidden_size, dropout)

        #self.edge_index = edge_index.to(self.device)
        #self.adj_label = adj_label.to(self.device).requires_grad_(True)
        #self.attrs = attrs.to(self.device).requires_grad_(True)
        
        self.label = label
        self.top_k_AS = None
        self.score = None
        self.contamination = 0.1
        self.threshold_ = None
        self.training = True
        self._adj_preped = None
        self._do_cache_adj_prep = True

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index, edge_weight = self._preprocess_adjacency_matrix(edge_index, x)
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)

        x = self.shared_encoder(x, edge_index, edge_weight)
        x_hat = self.attr_decoder(x, edge_index, edge_weight)
        struct_reconstructed = self.struct_decoder(x, edge_index, edge_weight)
        return struct_reconstructed, x_hat

    def fit(self, config: dict, new_edge_index, attrs, verbose: bool = False, top_k: int = 10):
        optimizer = torch.optim.Adam(self.parameters(), lr=config['model']['lr'])
        edge_index = new_edge_index
        edge_index = add_self_loops(edge_index)[0].to(self.device)
        print("Fitting on edge index of shape: ", edge_index.shape)

        
        for epoch in range(config['model']['epochs']):
            #adj, adj_label = prepare_adj_and_adj_label(edge_index=self.edge_index)
            #edge_index = to_edge_index(torch.sparse_coo_tensor(adj.nonzero(), adj.data, adj.shape))[0].to(self.device)
            #adj_label = torch.tensor(adj_label).to(self.device)
            #print(edge_index[0].shape)
            #adj_label = torch.tensor(adj_label).to(self.device)

            self.train()
            optimizer.zero_grad()
            # TODO: Normalize for every forward step
            A_hat, X_hat = self.forward(attrs, edge_index)
            #self.adj_label = to_dense_adj(self.edge_index)[0]
            #self.adj_label = self.adj_label + np.eye(self.adj_label.shape[0])
            loss, struct_loss, feat_loss = loss_func(to_dense_adj(edge_index)[0], A_hat, attrs, X_hat, config['model']['alpha'])
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"Epoch: {epoch:04d}, train_loss={loss.item():.5f}, "
                    f"train/struct_loss={struct_loss.item():.5f}, train/feat_loss={feat_loss.item():.5f}")

            if (epoch % 10 == 0 and verbose) or epoch == config['model']['epochs'] - 1:
                self.eval()
                A_hat, X_hat = self.forward(attrs, edge_index)
                loss, struct_loss, feat_loss = loss_func(to_dense_adj(edge_index)[0].to(self.device), A_hat, attrs, X_hat, config['model']['alpha'])
                self.score = loss.detach().cpu().numpy()
                #self.threshold_ = np.percentile(self.score, 100 * (1 - self.contamination))
                #pred = (self.score > self.threshold_)
                #print(pred)
                #print(self.label[33], self.label[65], self.label[88], self.label[89], self.label[90])

                print(f"Epoch: {epoch:04d}, Auc: {roc_auc_score(self.label.detach().cpu().numpy(), self.score)}")

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

        if self.training and self._adj_preped is not None:
            return self._adj_preped

        # set a timer and end of timer and calculate difference
        #start = time.perf_counter()
        adj = get_jaccard(
            torch.sparse.FloatTensor(
                edge_idx,
                torch.ones_like(edge_idx[0], dtype=torch.float32)
            ),
            x
        ).coalesce()
        #end = time.perf_counter()
        edge_idx, edge_weight = adj.indices(), adj.values()
        del adj

        if (
            self.training
            and self._do_cache_adj_prep
        ):
            print("CACHING")
            self._adj_preped = (edge_idx, edge_weight)
        
        #elapsed = end - start
        #print(f"Time difference: {elapsed:0.4f} seconds")
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

def prepare_adj_and_adj_label(edge_index):
    adj = to_dense_adj(edge_index)[0].detach().cpu().numpy()
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    
    #print("DENSE")
    #print(type(adj_norm))
    edge_index = to_edge_index(torch.sparse_coo_tensor(adj_norm.nonzero(), adj_norm.data, adj_norm.shape))
    #print(edge_index)
    #print(adj_norm)
    adj = adj + np.eye(adj.shape[0])
    return adj_norm, adj # adj and adj label

def load_anomaly_detection_dataset(dataset: Data, datadir: str = 'data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge_index = dataset.edge_index
    adj = to_dense_adj(edge_index)[0].detach().cpu().numpy()

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

    from torch_geometric.utils import to_torch_sparse_tensor, dense_to_sparse
    #edge_index = to_torch_sparse_tensor(dataset.edge_index.to(config['model']['device']))
    edge_index = dataset.edge_index.to(config['model']['device'])
    #edge_index = dense_to_sparse(torch.tensor(adj))[0].to(config['model']['device'])
    label = torch.Tensor(dataset.y.bool()).to(config['model']['device'])
    attrs = dataset.x.to(config['model']['device'])

    model = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                     device=config['model']['device'], edge_index=edge_index, adj_label=adj_label, attrs=attrs, label=label)
    model.to(config['model']['device'])
    model.fit(config, verbose=True, new_edge_index=edge_index, attrs=attrs)
    