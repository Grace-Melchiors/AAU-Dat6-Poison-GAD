from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from torch_geometric.utils import to_dense_adj

from pygod.detector import DOMINANT

from dominant_model import Dominant
#from dominant_utils import load_anomaly_detection_dataset
from pygod.utils import load_data

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

from torch_geometric.data import Data
def load_anomaly_detection_dataset(dataset, datadir='data'):
    # import dataset and extract its parts
    dataset = load_data("inj_cora")
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


def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)


    cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost

def train_dominant(args):
    adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)
    #print size of the four above variables formatted with their names
    print('label', label.shape)

    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    attrs = torch.FloatTensor(attrs)

    model = Dominant(feat_size = attrs.size(1), hidden_size = args.hidden_dim, dropout = args.dropout)

    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        attrs = attrs.to(device)
        model = model.cuda()
        
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        A_hat, X_hat = model(attrs, adj)
        print(f'Shape AHAT {A_hat.shape}, shape XHAT {X_hat.shape}')
        loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, args.alpha)
        l = torch.mean(loss)
        l.backward()
        optimizer.step()        
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))

        if epoch%10 == 0 or epoch == args.epoch - 1:
            model.eval()
            A_hat, X_hat = model(attrs, adj)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, args.alpha)
            score = loss.detach().cpu().numpy()
            print(f'Score size: {score.shape}')
            print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(label, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='BlogCatalog', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8, help='balance parameter')
    parser.add_argument('--device', default='cpu', type=str, help='cuda/cpu')

    args = parser.parse_args()

    train_dominant(args)