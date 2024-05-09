
import numpy as np
import scipy.sparse as sp
import torch
from torch.optim.optimizer import required
from torch.optim import Optimizer

import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
from typing import Tuple


# From https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/utils.py
def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# From https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense/pgd.py

class PGD(Optimizer):
    """Proximal gradient descent.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining parameter groups
    proxs : iterable
        iterable of proximal operators
    alpha : iterable
        iterable of coefficients for proximal gradient descent
    lr : float
        learning rate
    momentum : float
        momentum factor (default: 0)
    weight_decay : float
        weight decay (L2 penalty) (default: 0)
    dampening : float
        dampening for momentum (default: 0)

    """

    def __init__(self, params, proxs, alphas, lr=required, momentum=0, dampening=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)
        
        #Added such that it can be accessed in setstate, which previously gave errors
        self.proxs = proxs
        self.alphas = alphas
        #End of added

        super(PGD, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def __setstate__(self, state):
        super(PGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('proxs', self.proxs)
            group.setdefault('alphas', self.alphas)

    def step(self, delta=0, closure=None):
         for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            proxs = group['proxs']
            alphas = group['alphas']

            # apply the proximal operator to each parameter in a group
            for param in group['params']:
                for prox_operator, alpha in zip(proxs, alphas):
                    # param.data.add_(lr, -param.grad.data)
                    # param.data.add_(delta)
                    param.data = prox_operator(param.data, alpha=alpha*lr)


class ProxOperators():
    """Proximal Operators.
    """

    def __init__(self):
        self.nuclear_norm = None

    def prox_l1(self, data, alpha):
        """Proximal operator for l1 norm.
        """
        data = torch.mul(torch.sign(data), torch.clamp(torch.abs(data)-alpha, min=0))
        return data

    def prox_nuclear(self, data, alpha):
        """Proximal operator for nuclear norm (trace norm).
        """
        device = data.device
        U, S, V = np.linalg.svd(data.cpu())
        U, S, V = torch.FloatTensor(U).to(device), torch.FloatTensor(S).to(device), torch.FloatTensor(V).to(device)
        self.nuclear_norm = S.sum()
        # print("nuclear norm: %.4f" % self.nuclear_norm)

        diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

    def prox_nuclear_truncated_2(self, data, alpha, k=50):
        device = data.device
        import tensorly as tl
        tl.set_backend('pytorch')
        U, S, V = tl.truncated_svd(data.cpu(), n_eigenvecs=k)
        U, S, V = torch.FloatTensor(U).to(device), torch.FloatTensor(S).to(device), torch.FloatTensor(V).to(device)
        self.nuclear_norm = S.sum()
        # print("nuclear norm: %.4f" % self.nuclear_norm)

        S = torch.clamp(S-alpha, min=0)

        # diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        # U = torch.spmm(U, diag_S)
        # V = torch.matmul(U, V)

        # make diag_S sparse matrix
        indices = torch.tensor((range(0, len(S)), range(0, len(S)))).to(device)
        values = S
        diag_S = torch.sparse.FloatTensor(indices, values, torch.Size((len(S), len(S))))
        V = torch.spmm(diag_S, V)
        V = torch.matmul(U, V)
        return V

    def prox_nuclear_truncated(self, data, alpha, k=50):
        device = data.device
        indices = torch.nonzero(data).t()
        values = data[indices[0], indices[1]] # modify this based on dimensionality
        data_sparse = sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()))
        U, S, V = sp.linalg.svds(data_sparse, k=k)
        U, S, V = torch.FloatTensor(U).to(device), torch.FloatTensor(S).to(device), torch.FloatTensor(V).to(device)
        self.nuclear_norm = S.sum()
        diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

    def prox_nuclear_cuda(self, data, alpha):

        device = data.device
        U, S, V = torch.svd(data)
        # self.nuclear_norm = S.sum()
        # print(f"rank = {len(S.nonzero())}")
        self.nuclear_norm = S.sum()
        S = torch.clamp(S-alpha, min=0)
        indices = torch.tensor([range(0, U.shape[0]),range(0, U.shape[0])]).to(device)
        values = S
        diag_S = torch.sparse.FloatTensor(indices, values, torch.Size(U.shape))
        # diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        # print(f"rank_after = {len(diag_S.nonzero())}")
        V = torch.spmm(diag_S, V.t_())
        V = torch.matmul(U, V)
        return V

# From https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense/prognn.py

def loss_func(adj: torch.Tensor, A_hat: torch.Tensor, attrs: torch.Tensor, X_hat: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors
    return cost, structure_cost, attribute_cost

class ProGNN:
    """ ProGNN (Properties Graph Neural Network). See more details in Graph Structure Learning for Robust Graph Neural Networks, KDD 2020, https://arxiv.org/abs/2005.10203.

    Parameters
    ----------
    model:
        model: The backbone GNN model in ProGNN
    args:
        model configs
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
    See details in https://github.com/ChandlerBang/Pro-GNN.

    """

    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)

    def fit(self, features, adj, labels, top_k = 10, **kwargs):
        """Train Pro-GNN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format is torch.tensor
        labels :
            node labels
        top_k :
            the number of top k nodes to get
        """
        args = self.args

        # self.optimizer = optim.Adam(self.model.parameters(),
        #                        lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr) #Our other versions do not use weight decay


        estimator = EstimateAdj(adj, symmetric=args.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(),
                              momentum=0.9, lr=args.lr_adj)

        self.optimizer_l1 = PGD(estimator.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=args.lr_adj, alphas=[args.alpha])


        # warnings.warn("If you find the nuclear proximal operator runs too slow on Pubmed, you can  uncomment line 67-71 and use prox_nuclear_cuda to perform the proximal on gpu.")
        # if args.dataset == "pubmed":
        #     self.optimizer_nuclear = PGD(estimator.parameters(),
        #               proxs=[prox_operators.prox_nuclear_cuda],
        #               lr=args.lr_adj, alphas=[args.beta])
        # else:
        warnings.warn("If you find the nuclear proximal operator runs too slow, you can modify line 77 to use prox_operators.prox_nuclear_cuda instead of prox_operators.prox_nuclear to perform the proximal on GPU. See details in https://github.com/ChandlerBang/Pro-GNN/issues/1")
        self.optimizer_nuclear = PGD(estimator.parameters(),
                  proxs=[prox_operators.prox_nuclear],
                  lr=args.lr_adj, alphas=[args.beta])

        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            if args.only_gcn:
                self.train_DOMINANT(epoch, estimator.estimated_adj)
                # self.train_gcn(epoch, features, estimator.estimated_adj,
                #         labels, idx_train, idx_val)
            else:
                for i in range(int(args.outer_steps)):
                    self.train_adj(epoch, features, adj)

                for i in range(int(args.inner_steps)):
                    self.train_DOMINANT(epoch, estimator.estimated_adj)
                    # self.train_gcn(epoch, features, estimator.estimated_adj,
                    #         labels, idx_train, idx_val)

        # Get evals from DOMINANT model
        self.eval()
        A_hat, X_hat = self.forward(self.attrs, self.edge_index)
        loss, struct_loss, feat_loss = loss_func(self.adj_label, A_hat, self.attrs, X_hat, args.alpha)
        self.score = loss.detach().cpu().numpy()

        # Identify and store the IDs of the nodes with the top K highest anomaly scores
        
        # Convert the anomaly scores to a PyTorch tensor
        scores_tensor = torch.tensor(self.score)
        # Use torch.topk to find the top K scores and their indices
        topk_scores, topk_indices = torch.topk(scores_tensor, top_k, largest=True)
        # Convert the indices and scores to lists and store them
        top_k_AS_indices = topk_indices.tolist()
        top_k_AS_scores = topk_scores.tolist()
        self.model.top_k_AS = top_k_AS_indices
        # Print the node IDs and their corresponding anomaly scores
        if args.debug:
            print(f"Top {top_k} highest anomaly scores' node IDs and scores:")
            for idx, score in zip(top_k_AS_indices, top_k_AS_scores):
                print(f"Node ID: {idx}, Anomaly Score: {score}")

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(args)

        # Testing
        # print("picking the best model according to validation performance")
        # self.model.load_state_dict(self.weights)


    def train_DOMINANT(self, epoch, estimated_adj):
        args = self.args
        if args.debug:
            print("\n=== train_model ===")
        self.model.train()
        self.optimizer.zero_grad()


        edge_index = estimated_adj.clone().detach().to_sparse()

        A_hat, X_hat = self.model.forward(self.model.attrs, edge_index)

        loss, struct_loss, feat_loss = loss_func(self.model.adj_label, A_hat, self.model.attrs, X_hat, args.alpha)
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()
        if args.debug:
            print(f"Epoch: {epoch:04d}, train_loss={loss.item():.5f}, "
                f"train/struct_loss={struct_loss.item():.5f}, train/feat_loss={feat_loss.item():.5f}")

        
    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val):
        args = self.args
        estimator = self.estimator
        adj = estimator.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))



    def train_adj(self, epoch, features, adj):
        estimator = self.estimator
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        normalized_adj = estimator.normalize()

        if args.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        # output = self.model(features, normalized_adj) #Forward func
        # loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        # acc_train = accuracy(output[idx_train], labels[idx_train])

        edge_index = normalized_adj.clone().detach().to_sparse()

        A_hat, X_hat = self.model.forward(self.model.attrs, edge_index)
        loss, _, _ = loss_func(self.model.adj_label, A_hat, self.model.attrs, X_hat, args.alpha)
        loss_gcn = torch.mean(loss)


        loss_symmetric = torch.norm(estimator.estimated_adj \
                        - estimator.estimated_adj.t(), p="fro")

        loss_diffiential =  loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_smooth_feat + args.phi * loss_symmetric

        loss_diffiential.backward()

        self.optimizer_adj.step()
        loss_nuclear =  0 * loss_fro
        if args.beta != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = loss_fro \
                    + args.gamma * loss_gcn \
                    + args.alpha * loss_l1 \
                    + args.beta * loss_nuclear \
                    + args.phi * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(
                  estimator.estimated_adj.data, min=0, max=1))

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_symmetric: {:.4f}'.format(loss_symmetric.item()),
                      'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj-adj, 1).item()),
                      'loss_l1: {:.4f}'.format(loss_l1.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()),
                      'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))


    def test(self, features, labels, idx_test):
        """Evaluate the performance of ProGNN on test set
        """
        print("\t=== testing ===")
        self.model.eval()
        adj = self.best_graph
        if self.best_graph is None:
            adj = self.estimator.normalize()
        output = self.model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tTest set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())/2
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx




# Load own stuff
class Args: #Has comments from what pro-gnn recommends for their own GCN system
    def __init__(self, config):
        self.debug = False
        self.only_gcn = False
        self.no_cuda = False
        self.seed = 123
        self.lr = config['model']['lr'] #0.01
        self.weight_decay = 5e-4
        self.hidden = 16
        self.dropout = config['model']['dropout'] # 0.5
        self.dataset = 'cora'
        self.attack = 'meta'
        self.ptb_rate = 0.05
        self.epochs = config['model']['epochs'] #400
        self.alpha = config['model']['alpha'] #5e-4
        self.beta = 1.5
        self.gamma = 1
        self.lambda_ = 0
        self.phi = 0
        self.inner_steps = 2
        self.outer_steps = 1
        self.lr_adj = 0.01
        self.symmetric = False


prox_operators = ProxOperators()

def run_pro_gnn(data, config, dom_model):
    args = Args(config)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    estimator = ProGNN(data, config, args, dom_model)
    estimator.train()
    return estimator