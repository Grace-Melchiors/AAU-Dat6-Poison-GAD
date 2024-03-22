import torch.nn as nn
import torch.nn.functional as F
import torch
from dominant_layers import GraphConvolution
import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
from sklearn.metrics import roc_auc_score
import argparse
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_dense_adj


from dominant_model import Dominant
#from dominant_utils import load_anomaly_detection_dataset
from pygod.utils import load_data



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



# DOMINANT
class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x

class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x

class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout, device, adj, adj_label, attrs, label, lip_const):
        super(Dominant, self).__init__()
        
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)

        self.device = torch.device(args.device)
        print("Device is CUDA")
        self.adj = adj.to(self.device).requires_grad_(True)
        self.adj_label = adj_label.to(self.device).requires_grad_(True)
        self.attrs = attrs.to(self.device).requires_grad_(True)
        
        self.label = label
        self.lip_const = lip_const

    
    def forward(self, x, adj):
        # encode
        x = self.shared_encoder(x, adj)
        # decode feature matrix
        x_hat = self.attr_decoder(x, adj)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(x, adj)
        # return reconstructed matrices
        return struct_reconstructed, x_hat

    def fit(self, args):
        
        #if args.device == 'cuda':
                        #model = self.cuda()
        
        # Print type of self.device, self.adj, self.adj_label, self.attrs
        #print("self.device:", type(self.device), ", self.adj:", type(self.adj), ", self.adj_label:", type(self.adj_label), ", self.attrs:", type(self.attrs))
        optimizer = torch.optim.Adam(self.parameters(), lr = args.lr)

        for epoch in range(args.epoch):
            self.train()
            optimizer.zero_grad()
            A_hat, X_hat = self.forward(self.attrs, self.adj)
            print(f'Shape AHAT {A_hat.shape}, shape XHAT {X_hat.shape}')
            loss, struct_loss, feat_loss = loss_func(self.adj_label, A_hat, self.attrs, X_hat, args.alpha)
            
            # Lipschitz regularization
            lip_loss = self.lipschitz_loss()
            loss += args.lip_weight * lip_loss
            
            print(loss)
            l = torch.mean(loss)
            print(l)
            l.backward()
            optimizer.step()        
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))

            if epoch%10 == 0 or epoch == args.epoch - 1:
                self.eval()
                A_hat, X_hat = self.forward(self.attrs, self.adj)
                loss, struct_loss, feat_loss = loss_func(self.adj_label, A_hat, self.attrs, X_hat, args.alpha)
                score = loss.detach().cpu().numpy()
                print(f'Score size: {score.shape}')
                print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(self.label, score))
                
    def lipschitz_loss(self):
        lip_loss = 0.0
        for layer in self.modules():
            if isinstance(layer, GraphConvolution):
                weight = layer.weight
                lip_loss += torch.mean(torch.max(torch.sum(torch.abs(weight), dim=1) - self.lip_const, torch.tensor(0.0)))
        return lip_loss


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='BlogCatalog', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8, help='balance parameter')
    parser.add_argument('--device', default='cpu', type=str, help='cuda/cpu')
    parser.add_argument('--lip_const', type=float, default=1.0, help='Lipschitz constant')
    parser.add_argument('--lip_weight', type=float, default=0.1, help='Lipschitz regularization weight')

    args = parser.parse_args()

    #adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)

    #print size of the four above variables formatted with their names
    #print('label', label.shape)

    adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)
    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    attrs = torch.FloatTensor(attrs)

    model = Dominant(feat_size = attrs.size(1), hidden_size = args.hidden_dim, dropout = args.dropout, device = args.device, adj=adj, adj_label=adj_label, attrs=attrs, label=label, lip_const=args.lip_const)

    model.fit(args)