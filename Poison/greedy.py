# From https://github.com/zhuyulin-tony/BinarizedAttack/blob/main/src/Greedy.py

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch_sparse import SparseTensor

from Poison.base_classes import BasePoison

class multiple_AS(nn.Module):
    def __init__(self, target_lst, n_node, device):
        """
            target_lst (numpy.ndarray): The target list to be initialized.
            n_node (int): The number of nodes.
            device (str): The device to be used.

        """
        super().__init__()
        self.lst = target_lst
        self.n = n_node
        self.device = device
    
    def adjacency_matrix(self, tri):
        A = torch.sparse_coo_tensor(tri[:,:2].T, tri[:,2], size=[self.n,self.n]).to_dense()
        A = A + A.T - torch.diag(torch.diag(A)) # symmetric.
        return A
    
    def sparse_matrix_power(self, A, tau):
        A_sp = A.to_sparse()
        A_sp = torch.sparse_coo_tensor(A_sp.indices(), A_sp.values(), size=[self.n,self.n])
        return torch.sparse.mm(torch.sparse.mm(A_sp, A_sp), A_sp).to_dense()
    
    def extract_NE(self, A):    # Extract node and edge information based on adjecency matrix
        N = torch.sum(A, 1)
        E = torch.sum(A, 1) + 0.5 * torch.diag(self.sparse_matrix_power(A, 3)).T
        N = N.reshape(-1,1)
        E = E.reshape(-1,)
        return N, E
    
    def OLS_estimation(self, N, E):
        """
        OLS estimation function that calculates the Ordinary Least Squares estimate.
        
        Parameters:
            N (tensor): Input tensor for independent variable N
            E (tensor): Input tensor for dependent variable E
        
        Returns:
            tensor: Tensor result of the OLS estimation
        """
        logN = torch.log(N + 1e-20)
        logE = torch.log(E + 1e-20)
        logN1 = torch.cat((torch.ones((len(logN),1)).to(self.device), logN), 1)
        return torch.linalg.pinv(logN1) @ logE
        
    def forward(self, tri): # Calculate the loss function / How much the output deviates from expected least squares estimate
        A = self.adjacency_matrix(tri)
        N, E = self.extract_NE(A)
        theta = self.OLS_estimation(N, E)
        b = theta[0] # Intercept
        w = theta[1] # Coefficient
        tmp = 0.
        for i in range(len(self.lst)):
            tmp += (torch.exp(b) * (N[self.lst[i]]**w) - E[self.lst[i]])**2 # Accumulate squared difference between expected (b * N[i]**w) and actual E[i]
        return tmp
    
    def true_AS(self, tri): # Calculate the true anomaly score by using OLS (Page 3 https://arxiv.org/pdf/2106.09989.pdf)
        # Originally from https://www.cs.cmu.edu/~christos/courses/826.F11/CMU-ONLY/oddball.pdf
        # Or https://www.researchgate.net/profile/Leman-Akoglu/publication/220894884_OddBall_Spotting_Anomalies_in_Weighted_Graphs/links/0fcfd50b2ea00b30d2000000/OddBall-Spotting-Anomalies-in-Weighted-Graphs.pdf

        A = self.adjacency_matrix(tri)
        N, E = self.extract_NE(A)
        theta = self.OLS_estimation(N, E)
        b = theta[0] # Intercept
        w = theta[1] # Coefficient
        tmp = 0.
        for i in range(len(self.lst)):
            tmp += (torch.max(E[self.lst[i]],torch.exp(b)*(N[self.lst[i]]**w))\
                   /torch.min(E[self.lst[i]],torch.exp(b)*(N[self.lst[i]]**w)))*\
                    torch.log(torch.abs(E[self.lst[i]]-torch.exp(b)*(N[self.lst[i]]**w))+1)
        return tmp

def Poison_attack(model, triple, B, print_stats = False):
    triple_copy = triple.copy()
    triple_torch = Variable(torch.from_numpy(triple_copy), requires_grad = True) 
    AS = []
    perturb = []
    AS.append(model.true_AS(triple_torch).data.numpy()[0])
    if(print_stats): print('initial anomaly score:', model.true_AS(triple_torch).data.numpy()[0])
    
    for i in range(1,B+1):
        loss = model.forward(triple_torch)
        loss.backward()
        
        tmp = triple_torch.grad.data.numpy() # Get gradient of tensor with respect to data, stored in tmp


        grad = np.concatenate((triple_torch[:,:2].data.numpy(),tmp[:,2:]),1) # Concat edge descriptor with gradients
        

        v_grad = np.zeros((len(grad),3))

        for j in range(len(grad)):
            v_grad[j,0] = grad[j,0]
            v_grad[j,1] = grad[j,1]
            if triple_copy[j,2] == 0 and grad[j,2] < 0: # If no edge and gradient is negative
                v_grad[j,2] = grad[j,2]
            elif triple_copy[j,2] == 1 and grad[j,2] > 0: # If edge and gradient is positive
                v_grad[j,2] = grad[j,2]
            else:
                continue

        # Get indexes of sorted gradients in descending order [3,1,2]->[1,2,0]
        v_grad = v_grad[np.abs(v_grad[:,2]).argsort()] 
        

        # attack w.r.t gradient information.
        K = -1

        # Takes the edge with largest gradient by using neg K value(last k element)and finds the first that isn't already changed
        # Thusly changing the edge with the highest value
        while v_grad[K][:2].astype('int').tolist() in perturb:
            K -= 1
            
            
        # do not delete edge from singleton.
        while v_grad[int(K)][2] > 0 and \
             (model.adjacency_matrix(triple_torch).data.numpy()[int(v_grad[int(K)][0])].sum() <= 1 or \
              model.adjacency_matrix(triple_torch).data.numpy()[int(v_grad[int(K)][1]) ].sum() <= 1):
            K -= 1
        
        target_grad = v_grad[int(K)] #Picks edge to target

        # Get index of target in triple
        target_index = np.where(np.all((triple[:,:2] == target_grad[:2]), axis = 1) == True)[0][0]

        # Update representation of adjacency matrix (triple_torch)
        triple_copy[target_index,2] -= np.sign(target_grad[2])
        triple_torch = Variable(torch.from_numpy(triple_copy), requires_grad = True)

        # Add perturb to list of perturbs
        perturb.append([int(target_grad[0]),int(target_grad[1])])

        # Get and save updated anomaly score
        true_AScore = model.true_AS(triple_torch).data.numpy()[0] 
        AS.append(true_AScore)
        if(print_stats): print('iter', i, 'anomaly score:', true_AScore)

    AS = np.array(AS)    

    sparse_tensor = triple_torch.to_sparse()

    return triple_torch, AS, perturb

from pygod.detector import DOMINANT
from Utils.graph_utils import prepare_graph, adj_matrix_sparse_coo_to_dense
from Utils.experiment_results import Experiment
import torch
from pygod.utils import load_data
import copy
from typing import Tuple, List, Any


import sys

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)

class Greedy_Poison_Class(BasePoison):
    def __init__(self, data, budget=0, target_node_indexes= []):
        super().__init__(data, budget, target_node_indexes)

    def poison_data(self, print_stats = False):
        return inject_greedy_poison(self.data, self.target_node_indexes, budget = self.budget, print_stats = print_stats)


def inject_greedy_poison(data, target_node_lst, seed = 0, budget = 35, print_stats = False):
    """
    Injects a greedy poison with Oddball Anomaly Score into the given graph data using the specified target nodes, seed, and budget.
    
    Parameters:
        data (torch_geometric.data.Data): The graph data to be poisoned.
        target_node_lst (array[int): The list of target nodes to poison.
        seed (int, optional): The seed value for random number generation. Defaults to 0.
        budget (int, optional): The budget for the number of edge modifications. Defaults to 35.
        print_stats (bool, optional): Whether to print statistics during the poisoning process. Defaults to False.
        
    Returns:
        torch_geometric.data.Data: The poisoned graph data.
    """

    target_node_lst = np.array(target_node_lst) # Convert to numpy for compatibility

   
    if(print_stats): print("Create poison compatible adjacency matrix...")

    _, adj, _ = prepare_graph(data) #Get adjacency matrix

    amount_of_nodes = data.num_nodes

    # 'triple' is a list that will store the perturbed triples during the poisoning process.
    # Each triple represents an edge modification in the form of (node1, node2, edge_label).

    dense_adj = adj.to_dense()  #Fill in zeroes where there are no edges
    
    triple = []
    for i in range(amount_of_nodes):
        for j in range(i + 1, amount_of_nodes):
            triple.append([i, j, dense_adj[i,j]])  #Fill with 0, then insert actual after

    triple = np.array(triple)

    if(print_stats): print("Making model...")
    model = multiple_AS(target_lst = target_node_lst, n_node = amount_of_nodes, device = 'cpu')

    if(print_stats): print("Starting attack...")

    adj_adversary, AS, list_of_perturbs = Poison_attack(model, triple, budget, print_stats = print_stats)

    if(print_stats): print("Converting to torch.geometric.data.Data...")

    # Create Edge Index'
    edge_index = torch.tensor([[],[]])

    # Transpose poisoned adj to make shape compatible
    transposed_adj_adversary = torch.transpose(adj_adversary, 0, 1)

    for i in range(len(adj_adversary)):
        if(adj_adversary[i][2] != 0):   #If edge value is not 0 (no edge)
            #Add edge to edge index, choosing first 2 elements (edges), and then the ith edge
            edge_index = torch.cat((edge_index, transposed_adj_adversary[:2, i:i+1]), -1)
            # Dataset uses edges both ways so add reverse edge as well
            edge_index = torch.cat((edge_index, torch.flip(transposed_adj_adversary[:2, i:i+1], dims=[0])), -1)
    

    # Make new data object with new edge index
    data_after_poison = copy.deepcopy(data)
    edge_index = edge_index.type(torch.int64)
    data_after_poison.edge_index = edge_index

    # Return poisoned data
    return data_after_poison




def run_greedy(budget = 2) -> Tuple[Experiment, Experiment, List[int], Any]:
    #data = Planetoid("./data/Cora", "Cora", transform=T.NormalizeFeatures())[0]
    data = load_data("inj_cora")
    y_binary: List[int] = data.y.bool()
    print(y_binary)

    detector = DOMINANT(hid_dim=64, num_layers=4, epoch=100)
    detector.fit(data)

    pred, score, prob, conf = detector.predict(data,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)
    experiment_before_poison = Experiment(data=data, pred=pred, prob=prob, score=score, conf=conf)

    _, adj, _ = prepare_graph(data)
    
 
    amount_of_nodes = data.num_nodes

    # 'triple' is a list that will store the perturbed triples during the poisoning process.
    # Each triple represents an edge modification in the form of (node1, node2, edge_label).

    dense_adj = adj.to_dense()  #Fill in zeroes where there are no edges


    print("Create poison compatible adjacency matrix...")
    triple = []
    for i in range(amount_of_nodes):
        for j in range(i + 1, amount_of_nodes):
            triple.append([i, j, dense_adj[i,j]])  #Fill with 0, then insert actual after

    triple = np.array(triple)

    # These are the nodes we try reduce the active subnetwork score for (disguising anonomaly nodes)
    target_node_lst = np.array([2, 3, 5])

    print("Making model...")
    model = multiple_AS(target_lst = target_node_lst, n_node = amount_of_nodes, device = 'cpu')
    budget = 2  # The amount of edges to change

    print("Starting attack...")

    adj_adversary, _, _ = Poison_attack(model, triple, budget)

    print("Converting to compatible tensor...")

    # Create Edge Index'
    edge_index = torch.tensor([[],[]])

    # Transpose it to make shape compatible
    transposed_adj_adversary = torch.transpose(adj_adversary, 0, 1)

    for i in range(len(adj_adversary)):
        if(adj_adversary[i][2] != 0):   #If edge value is not 0 (no edge)
            #Add edge to edge index, choosing first 2 elements (edges), and then the ith edge
            edge_index = torch.cat((edge_index, transposed_adj_adversary[:2, i:i+1]), -1)
            # Dataset uses edges both ways so add reverse edge as well
            edge_index = torch.cat((edge_index, torch.flip(transposed_adj_adversary[:2, i:i+1], dims=[0])), -1)
    

    # Make new data object with new edge index
    data_after_poison = copy.deepcopy(data)
    edge_index = edge_index.type(torch.int64)
    data_after_poison.edge_index = edge_index

    print("Running model on poisoned data...")

    detector_poisoned = DOMINANT(hid_dim=64, num_layers=4, epoch=100)
    detector_poisoned.fit(data_after_poison)

    pred_after, score_after, prob_after, conf_after = detector_poisoned.predict(data_after_poison,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)
    experiment_after_poison = Experiment(data=data_after_poison, pred=pred_after, score=score_after, prob=prob_after, conf=conf_after)
    
    
    print("auc before poison:")
    print(roc_auc_score(y_binary, score))
    print("auc after poison:")
    print(roc_auc_score(y_binary, score_after))
    #print(roc_auc_score(data_inj.y.detach().numpy(), score_inj.detach.numpy()))

    
    return experiment_before_poison, experiment_after_poison #, node_idxs, y_binary







