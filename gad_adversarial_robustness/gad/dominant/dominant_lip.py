import torch.nn as nn
import torch.nn.functional as F
import torch
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
from gad_adversarial_robustness.utils.graph_utils import load_anomaly_detection_dataset
from pygod.utils import load_data
from torch import autograd


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

class DominantLIP(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout, device, adj, adj_label, attrs, label):
        super(DominantLIP, self).__init__()
        
        self.device = device
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)

        self.device = torch.device(self.device)
        # This adjacency matrix is already normalized
        self.adj = adj.to(self.device).requires_grad_(True)
        self.adj_label = adj_label.to(self.device).requires_grad_(True)
        self.attrs = attrs.to(self.device).requires_grad_(True)
        
        self.label = label

    from torch_geometric.data import Data

    
    def forward(self, x, adj):
        # encode
        x = self.shared_encoder(x, adj)
        # decode feature matrix
        x_hat = self.attr_decoder(x, adj)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(x, adj)
        # return reconstructed matrices
        return struct_reconstructed, x_hat
    
    def lip_reg(self, features: torch.Tensor, adj, idx_train = None):
        lip_mat = []
        input = features.detach().clone()
        input.to_dense()
        input.requires_grad_(True)
        print("Forward Pass")
        output = self.forward(input, adj)[0]
        # Creates a zeros 

        for i in range(output.shape[1]):
            v = torch.zeros_like(output)
            # Create x tensors each with 1's in different columns.
            v[:, i] = 1
            print("Autograd")
            gradients = autograd.grad(outputs=output, inputs=input, grad_outputs=v,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            #gradients = gradients[idx_train]  
            grad_norm = torch.norm(gradients, dim=1).unsqueeze(dim=1)
            print("Appending")
            lip_mat.append(grad_norm)
        
        input.requires_grad_(False)
        # Concatenate all the matrices in lip_mat along the 1st dim (cols) 
        print("Concat")
        lip_concat = torch.cat(lip_mat, dim=1)
        lip_con_norm = torch.norm(lip_concat, dim=1)
        lip_loss = torch.max(lip_con_norm)
        return lip_loss



    def fit(self, gamma, args):
        
        #if args.device == 'cuda':
                        #model = self.cuda()
        
        # Print type of self.device, self.adj, self.adj_label, self.attrs
        #print("self.device:", type(self.device), ", self.adj:", type(self.adj), ", self.adj_label:", type(self.adj_label), ", self.attrs:", type(self.attrs))
        optimizer = torch.optim.Adam(self.parameters(), lr = args.lr)

        for epoch in range(args.epoch):
            self.train()
            optimizer.zero_grad()
            A_hat, X_hat = self.forward(self.attrs, self.adj)
            loss, struct_loss, feat_loss = loss_func(self.adj_label, A_hat, self.attrs, X_hat, args.alpha)
            l = torch.mean(loss)
            l.backward()
            optimizer.step()        
            #print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))

            if epoch%10 == 0 or epoch == args.epoch - 1:
                self.eval()
                A_hat, X_hat = self.forward(self.attrs, self.adj)
                loss, struct_loss, feat_loss = loss_func(self.adj_label, A_hat, self.attrs, X_hat, args.alpha)
                # We add a term for the lipschitz regularization to the loss

                loss  = loss + gamma * self.lip_reg(self.attrs, self.adj)
                score = loss.detach().cpu().numpy()
                #print(f'Score size: {score.shape}')
                print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(self.label, score))

                """
                indices = np.where(self.label == 1)[0]

                # Filter label and score arrays to include only nodes with label == 1
                filtered_label = self.label[indices]
                filtered_score = score[indices]

                threshold = 0.8

                # Convert filtered_score to predicted labels based on the threshold
                predicted_labels = (filtered_score >= threshold).astype(int)

                from sklearn.metrics import accuracy_score

                # Calculate accuracy

                accuracy = accuracy_score(filtered_label, predicted_labels)
                print(f'Epoch {epoch} Accuracy: {accuracy}')

                #print("Epoch:", '%04d' % (epoch) 'Accuracy:', accuracy)
                """



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()



# This loss function is different than the one in dominant.py
# We modify the loss function we add a regularization term for Lipschitz.
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

    args = parser.parse_args()
    print(f'args: {args}')

    #adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)

    #print size of the four above variables formatted with their names
    #print('label', label.shape)

    dataset = load_data("inj_cora")
    adj, attrs, poison_label, adj_label = load_anomaly_detection_dataset(dataset)
    poison_adj = torch.FloatTensor(adj)
    poison_adj_label = torch.FloatTensor(adj_label)
    poison_attrs = torch.FloatTensor(attrs)


    #model = DominantLIP(feat_size = attrs.size(1), hidden_size = args.hidden_dim, dropout = args.dropout, device = args.device, adj=adj, adj_label=adj_label, attrs=attrs, label=label)
    print("DOMINANT_LIP ACCURACY: ")
    dominant_lip = DominantLIP(feat_size = poison_attrs.size(1), hidden_size = args.hidden_dim, dropout = args.dropout, device = args.device, adj=poison_adj, adj_label=poison_adj_label, attrs=poison_attrs, label=poison_label)
    dominant_lip.fit(0.001, args)

    dominant_lip.fit(args)