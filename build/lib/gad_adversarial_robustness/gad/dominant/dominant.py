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
from typing import Tuple
from pygod.utils import load_data

class Encoder(nn.Module):
    def __init__(self, nfeat: int, nhid: int, dropout: float):
        super(Encoder, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, edge_index))
        return x


class AttributeDecoder(nn.Module):
    def __init__(self, nfeat: int, nhid: int, dropout: float):
        super(AttributeDecoder, self).__init__()
        self.gc1 = GCNConv(nhid, nhid)
        self.gc2 = GCNConv(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, edge_index))
        return x


class StructureDecoder(nn.Module):
    def __init__(self, nhid: int, dropout: float):
        super(StructureDecoder, self).__init__()
        self.gc1 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, edge_index))
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

        self.edge_index = edge_index.to(self.device)
        self.adj_label = adj_label.to(self.device).requires_grad_(True)
        self.attrs = attrs.to(self.device).requires_grad_(True)
        self.label = label
        self.score = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
                self.score = loss.detach().cpu().numpy()
                print(f"Epoch: {epoch:04d}, Auc: {roc_auc_score(self.label, self.score)}")


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

    dataset = load_data("inj_cora")
    adj, attrs, label, adj_label = load_anomaly_detection_dataset(dataset)
    edge_index = torch.LongTensor(np.array(sp.coo_matrix(adj).nonzero()))
    adj_label = torch.FloatTensor(adj_label)
    attrs = torch.FloatTensor(attrs)

    model = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                     device=config['model']['device'], edge_index=edge_index, adj_label=adj_label, attrs=attrs, label=label)
    model.fit(config, verbose=True)