# From https://github.com/zhuyulin-tony/BinarizedAttack/blob/main/src/Greedy.py

import time
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.utils import to_edge_index, dense_to_sparse, to_dense_adj
from copy import deepcopy

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
#from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix


from gad_adversarial_robustness.poison.base_classes import BasePoison

from gad_adversarial_robustness.gad.dominant.dominant_cuda import Dominant 
from gad_adversarial_robustness.utils.graph_utils import load_anomaly_detection_dataset
from torch_geometric.utils.convert import from_scipy_sparse_matrix

def accuracy(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    correct = torch.sum(y_true == y_pred).item()
    total = len(y_true)
    return correct / total

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
        N = N.reshape(-1,1).to(self.device)
        E = E.reshape(-1,).to(self.device)
        return N, E
    
    def OLS_estimation(self, N, E):
        """
        OLS estimation function that calculates the Ordinary Least Squares estimate.
        
        Parameters:
            N (tensor): Input tensor for independent variable N (node)
            E (tensor): Input tensor for dependent variable E (edge)
        
        Returns:
            tensor: Tensor result of the OLS estimation
        """
        logN = torch.log(N + 1e-20).to(self.device)
        logE = torch.log(E + 1e-20).to(self.device)
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

def update_adj_matrix_with_perturb(adj_matrix, perturb):
    """
        A faster way of converting perturbations to the edge_data

        Parameters: 
        - adj_matrix: The adjacency matrix in sparse
        - perturb: The perturbations to be added

        Returns:
        - adj_matrix: The updated adjacency matrix in sparse
    """
    adj_matrix = to_dense_adj(adj_matrix)
    #adj_matrix = adj_matrix.to_dense()
    if adj_matrix.ndim == 3:
        adj_matrix = adj_matrix[0]
    
    print("DENSE SHAPE:")
    print(adj_matrix[0].shape)

    for change in perturb:
        print(f'Change: {change}')
        adj_matrix[change[0], change[1]] = change[2]
        adj_matrix[change[1], change[0]] = change[2]
    

    print(adj_matrix.shape)
    adj_matrix = dense_to_sparse(adj_matrix)[0]
    print(adj_matrix.shape)
    return adj_matrix


        
def update_edge_data_with_perturb(edge_data, perturb):
    """
        A faster way of converting perturbations to the edge_data

        Parameters: 
        - edge_data: The edge_data to be updated in pytorch format
        - perturb: The perturbations to be added

        Returns:
        - edge_data: The updated edge_data
    """
    
    for change in perturb:
        if (change[2] == 1):    #If to add edge
            edge_data = torch.cat((edge_data, torch.tensor([[change[0], change[1]], [change[1], change[0]]])), -1)
        else:   #Delete edge

            for i in reversed(range(edge_data.size(dim = 1))):   #Cycle through all edges in reverse, if matches delete
                if ((edge_data[0][i] == change[0] and edge_data[1][i] == change[1]) or
                    (edge_data[0][i] == change[1] and edge_data[1][i] == change[0])):
                    edge_data = torch.cat((edge_data[:, :i], edge_data[:, i+1:]), axis = 1)
        
    return edge_data



def update_edge_index(edge_index, changes, device):
    updated_edge_index = edge_index.clone().to(device)
    
    for change in changes:
        source, target, weight = change
        
        if weight == 0:  # Add edge
            new_edge = torch.tensor([[source, target], [target, source]], dtype=torch.long, device=device)
            updated_edge_index = torch.cat([updated_edge_index, new_edge], dim=1)
        elif weight == 1.0:  # Remove edge
            mask = ~(((updated_edge_index[0] == source) & (updated_edge_index[1] == target)) | ((updated_edge_index[0] == target) & (updated_edge_index[1] == source)))
            updated_edge_index = updated_edge_index[:, mask]

    return updated_edge_index


def target_node_mask(target_list, tuple_list):
    """
        Takes a targetlist, and a tensor list. Returns a numpy array
    """
    new_list = []
    for index in target_list:
        new_list.append(tuple_list[index])
    
    return new_list
def get_DOMINANT_eval_values(model_obj, config, target_list, perturb, dom_params):
    """
        parameters:
        - model: The DOMINANT model
        - config: The config of the DOMINANT model
        - target_list: List of target nodes
        - perturb: perturbations

        returns:
        - AS_DOM: List of the anomaly score according to DOMINANT
        - AUC_DOM: AUC value according to DOMINANT
        - ACC_DOM: AUC value only considering target nodes, according to DOMINANT
    """

    model = model_obj(**dom_params)

    #deepcopy_model = deepcopy(model)
    #torch.save(deepcopy_model.state_dict(), 'model.pt')

    #model.edge_index = update_edge_data_with_perturb(model.edge_index, perturb)
    #start_time = time.time()
    
    #copy = model.edge_index
    new_edge_index = update_edge_index(dom_params['edge_index'], perturb, config['model']['device']).to(config['model']['device'])
    #model.edge_index = new_edge_index 
    #print(copy.shape, model.edge_index.shape, perturb)
    print(f"PERTURBATIONS: {perturb}")
    #diff = model.edge_index - copy
    #print(f'CHANGE MADE: {diff}')
    #end_time = time.time()
    #runtime = end_time - start_time
    #print(f'runtime: {runtime} seconds')


    
    model.to(config['model']['device'])
    model.fit(config, verbose=False, new_edge_index = new_edge_index, attrs = dom_params['attrs'])

    last_feat_loss = model.last_feat_loss
    last_struct_loss = model.last_struct_loss

    target_nodes_as = target_node_mask(target_list=target_list, tuple_list=model.score)
    #print("===========================")
    #print("All target nodes as:")
    #for node in target_nodes_as:
    #    print(node)
    #print("===========================")
    AS_DOM = np.sum(target_nodes_as)
    #AS_DOM = np.sum(model.score)
    AUC_DOM = roc_auc_score(model.label.detach().cpu().numpy(), model.score)
    print(f'{model.__class__.__name__}: AUC: {AUC_DOM} AS: {AS_DOM}')
    ACC_DOM = accuracy(model.label.detach().cpu().numpy(), model.score)
    #ACC_DOM = roc_auc_score(target_node_mask(model.label, target_list), target_node_mask(model.score, target_list))

    #model.load_state_dict(torch.load('model.pt'))

    return AS_DOM, AUC_DOM, ACC_DOM, target_nodes_as, last_feat_loss, last_struct_loss

def greedy_attack_with_statistics_multi(model, triple, DOMINANT_model_1, dom_params, config, target_list, B, CPI = 1, print_stats = False, DOMINANT_model_2 = None,  DOMINANT_model_3 = None, DOMINANT_model_4 = None):
    """
        Parameters: 
        - model: The surrogate model
        - triple: The edge_data to be posioned in triple form
        - DOMINANT_model_1: The first DOMINANT model
        - DOMINANT_model_2: The second DOMINANT model
        - config: The config of the DOMINANT model
        - target_list: List of target nodes
        - B: The number of perturbations
        - CPI: The number of perturbations per iteration
        - print_stats: Whether to print the anomaly score and changed edge after each perturbation

        Returbs:
        - triple: The poisoned edge_data
        - AS: List of the anomaly score after each perturbation according to surrogate model
        - AS_DOM: List of anomaly score after each perturbation according to DOMINANT
        - AUC_DOM: AUC value after each perturbation according to DOMINANT
        - ACC_DOM: Accuracy of the predicting the target nodes
        - perturb: List of the changed edges
        - edge_index: The edge_index of the poisoned graph, converted to torch format
    """

    triple_copy = triple.copy()
    # print(f'triple copy type: {type(triple_copy)}')
    triple_torch = Variable(torch.from_numpy(triple_copy), requires_grad = True) 

    AS = []

    ### Depends on number of models

    AS_DOM = [[]]
    AUC_DOM = [[]]
    ACC_DOM = [[]]
    LAST_FEAT_LOSS = [[]]
    LAST_STRUCT_LOSS = [[]]
    CHANGE_IN_AS_TARGET_NODE_AS = [[]]

    if DOMINANT_model_2 is not None:
        # Append empty array to 6 above arrays
        AS_DOM.append([])
        AUC_DOM.append([])
        ACC_DOM.append([])
        LAST_FEAT_LOSS.append([])
        LAST_STRUCT_LOSS.append([])
        CHANGE_IN_AS_TARGET_NODE_AS.append([])

    if DOMINANT_model_3 is not None:
        # Append empty array to 6 above arrays
        AS_DOM.append([])
        AUC_DOM.append([])
        ACC_DOM.append([])
        LAST_FEAT_LOSS.append([])
        LAST_STRUCT_LOSS.append([])
        CHANGE_IN_AS_TARGET_NODE_AS.append([])

    if DOMINANT_model_4 is not None:
        # Append empty array to 6 above arrays
        AS_DOM.append([])
        AUC_DOM.append([])
        ACC_DOM.append([])
        LAST_FEAT_LOSS.append([])
        LAST_STRUCT_LOSS.append([])
        CHANGE_IN_AS_TARGET_NODE_AS.append([])

    ###

    perturb = []
    AS.append(model.true_AS(triple_torch).data.detach().cpu().numpy()[0])

    AS_DOM_temp_1, AUC_DOM_temp_1, ACC_DOM_temp_1, target_nodes_as_1, last_feat_loss_1, last_struct_loss_1 = get_DOMINANT_eval_values(DOMINANT_model_1, config, target_list, perturb, dom_params)
    insert_count = 0

    LAST_FEAT_LOSS[insert_count].append(last_feat_loss_1)
    LAST_STRUCT_LOSS[insert_count].append(last_struct_loss_1)
    CHANGE_IN_AS_TARGET_NODE_AS[insert_count].append(target_nodes_as_1)
    AS_DOM[insert_count].append(AS_DOM_temp_1)
    AUC_DOM[insert_count].append(AUC_DOM_temp_1)
    ACC_DOM[insert_count].append(ACC_DOM_temp_1)
    insert_count +=1


    if DOMINANT_model_2 is not None:
        AS_DOM_temp_2, AUC_DOM_temp_2, ACC_DOM_temp_2, target_nodes_as_2, last_feat_loss_2, last_struct_loss_2 = get_DOMINANT_eval_values(DOMINANT_model_2, config, target_list, perturb, dom_params)
        LAST_FEAT_LOSS[insert_count].append(last_feat_loss_2)
        LAST_STRUCT_LOSS[insert_count].append(last_struct_loss_2)
        CHANGE_IN_AS_TARGET_NODE_AS[insert_count].append(target_nodes_as_2)
        AS_DOM[insert_count].append(AS_DOM_temp_2)
        AUC_DOM[insert_count].append(AUC_DOM_temp_2)
        ACC_DOM[insert_count].append(ACC_DOM_temp_2)
        insert_count += 1
    if DOMINANT_model_3 is not None:
        AS_DOM_temp_3, AUC_DOM_temp_3, ACC_DOM_temp_3, target_nodes_as_3, last_feat_loss_3, last_struct_loss_3 = get_DOMINANT_eval_values(DOMINANT_model_3, config, target_list, perturb, dom_params)
        LAST_FEAT_LOSS[insert_count].append(last_feat_loss_3)
        LAST_STRUCT_LOSS[insert_count].append(last_struct_loss_3)
        CHANGE_IN_AS_TARGET_NODE_AS[insert_count].append(target_nodes_as_3)
        AS_DOM[insert_count].append(AS_DOM_temp_3)
        AUC_DOM[insert_count].append(AUC_DOM_temp_3)
        ACC_DOM[insert_count].append(ACC_DOM_temp_3)
        insert_count += 1
    if DOMINANT_model_4 is not None:
        AS_DOM_temp_4, AUC_DOM_temp_4, ACC_DOM_temp_4, target_nodes_as_4, last_feat_loss_4, last_struct_loss_4 = get_DOMINANT_eval_values(DOMINANT_model_4, config, target_list, perturb, dom_params)
        LAST_FEAT_LOSS[insert_count].append(last_feat_loss_4)
        LAST_STRUCT_LOSS[insert_count].append(last_struct_loss_4)
        CHANGE_IN_AS_TARGET_NODE_AS[insert_count].append(target_nodes_as_4)
        AS_DOM[insert_count].append(AS_DOM_temp_4)
        AUC_DOM[insert_count].append(AUC_DOM_temp_4)
        ACC_DOM[insert_count].append(ACC_DOM_temp_4)
        insert_count += 1





    if(print_stats): print('initial anomaly score:', model.true_AS(triple_torch).data.detach().cpu().numpy()[0])
    
    #i = 0
    count = 0
    for i in range(1, B+1):   #While we have not reached the maximum number of perturbations
        count += 1
        print("====================")
        print("Perturbation number:", count)
        
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
        #slice = [perturb[i][0:2] for i in range(len(perturb))]
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
        #triple_torch = Variable(torch.from_numpy(triple_copy), requires_grad = True)

        # Add perturb to list of perturbs
        perturb.append([int(target_grad[0]),int(target_grad[1]), int(0 < target_grad[2])]) 

        # Get and save updated scores and values
        true_AScore = model.true_AS(triple_torch).data.detach().cpu().numpy()[0] 
        AS.append(true_AScore)

        # Here
        AS_DOM_temp_1, AUC_DOM_temp_1, ACC_DOM_temp_1, target_nodes_as_1, last_feat_loss_1, last_struct_loss_1 = get_DOMINANT_eval_values(DOMINANT_model_1, config, target_list, perturb, dom_params)
        insert_count = 0

        LAST_FEAT_LOSS[insert_count].append(last_feat_loss_1)
        LAST_STRUCT_LOSS[insert_count].append(last_struct_loss_1)
        CHANGE_IN_AS_TARGET_NODE_AS[insert_count].append(target_nodes_as_1)
        AS_DOM[insert_count].append(AS_DOM_temp_1)
        AUC_DOM[insert_count].append(AUC_DOM_temp_1)
        ACC_DOM[insert_count].append(ACC_DOM_temp_1)
        insert_count +=1


        if DOMINANT_model_2 is not None:
            AS_DOM_temp_2, AUC_DOM_temp_2, ACC_DOM_temp_2, target_nodes_as_2, last_feat_loss_2, last_struct_loss_2 = get_DOMINANT_eval_values(DOMINANT_model_2, config, target_list, perturb, dom_params)
            LAST_FEAT_LOSS[insert_count].append(last_feat_loss_2)
            LAST_STRUCT_LOSS[insert_count].append(last_struct_loss_2)
            CHANGE_IN_AS_TARGET_NODE_AS[insert_count].append(target_nodes_as_2)
            AS_DOM[insert_count].append(AS_DOM_temp_2)
            AUC_DOM[insert_count].append(AUC_DOM_temp_2)
            ACC_DOM[insert_count].append(ACC_DOM_temp_2)
            insert_count += 1
        if DOMINANT_model_3 is not None:
            AS_DOM_temp_3, AUC_DOM_temp_3, ACC_DOM_temp_3, target_nodes_as_3, last_feat_loss_3, last_struct_loss_3 = get_DOMINANT_eval_values(DOMINANT_model_3, config, target_list, perturb, dom_params)
            LAST_FEAT_LOSS[insert_count].append(last_feat_loss_3)
            LAST_STRUCT_LOSS[insert_count].append(last_struct_loss_3)
            CHANGE_IN_AS_TARGET_NODE_AS[insert_count].append(target_nodes_as_3)
            AS_DOM[insert_count].append(AS_DOM_temp_3)
            AUC_DOM[insert_count].append(AUC_DOM_temp_3)
            ACC_DOM[insert_count].append(ACC_DOM_temp_3)
            insert_count += 1
        if DOMINANT_model_4 is not None:
            AS_DOM_temp_4, AUC_DOM_temp_4, ACC_DOM_temp_4, target_nodes_as_4, last_feat_loss_4, last_struct_loss_4 = get_DOMINANT_eval_values(DOMINANT_model_4, config, target_list, perturb, dom_params)
            LAST_FEAT_LOSS[insert_count].append(last_feat_loss_4)
            LAST_STRUCT_LOSS[insert_count].append(last_struct_loss_4)
            CHANGE_IN_AS_TARGET_NODE_AS[insert_count].append(target_nodes_as_4)
            AS_DOM[insert_count].append(AS_DOM_temp_4)
            AUC_DOM[insert_count].append(AUC_DOM_temp_4)
            ACC_DOM[insert_count].append(ACC_DOM_temp_4)
            insert_count += 1



        
        if(print_stats): 
            print("Iteration: ", i)
            print('DOMINANT (regular): Anomaly score:', true_AScore, '--- DOM anomaly score:', AS_DOM_temp_1, 
                                '--- DOM AUC:', AUC_DOM_temp_1, '--- TARGET DOM ACC:', ACC_DOM_temp_1)
            if DOMINANT_model_2 is not None:
                print('DOMINANT w/ SM: Anomaly score:', true_AScore, '--- DOM anomaly score:', AS_DOM_temp_2, 
                                    '--- DOM AUC:', AUC_DOM_temp_2, '--- TARGET DOM ACC:', ACC_DOM_temp_2)
            if DOMINANT_model_3 is not None:
                print('DOMINANT w/ Jaccard Anomaly score:', true_AScore, '--- DOM anomaly score:', AS_DOM_temp_3, 
                                    '--- DOM AUC:', AUC_DOM_temp_3, '--- TARGET DOM ACC:', ACC_DOM_temp_3)
            if DOMINANT_model_4 is not None:
                print('DOMINANT w/ Jaccard and SM: Anomaly score:', true_AScore, '--- DOM anomaly score:', AS_DOM_temp_4, 
                                    '--- DOM AUC:', AUC_DOM_temp_4, '--- TARGET DOM ACC:', ACC_DOM_temp_4)
    AS = np.array(AS)    


    edge_index = update_edge_index(dom_params['edge_index'], perturb, config['model']['device']).to(config['model']['device'])
    print("Final edge index made.")
    #edge_index = update_edge_index(DOMINANT_model.edge_index, perturb)
    #edge_index = [] 
    #edge_index.append(update_adj_matrix_with_perturb(DOMINANT_model_1.edge_index, perturb))
    #edge_index.append(update_adj_matrix_with_perturb(DOMINANT_model_2.edge_index, perturb))

    return triple_torch, AS, AS_DOM, AUC_DOM, ACC_DOM, perturb, edge_index, CHANGE_IN_AS_TARGET_NODE_AS, LAST_FEAT_LOSS, LAST_STRUCT_LOSS



def greedy_attack_with_statistics(model, triple, DOMINANT_model, config, target_list, B, CPI = 1, print_stats = False):
    """
        Parameters: 
        - model: The surrogate model
        - triple: The edge_data to be posioned in triple form
        - DOMINANT_model: The DOMINANT model
        - config: The config of the DOMINANT model
        - target_list: List of target nodes
        - B: The number of perturbations
        - CPI: The number of perturbations per iteration
        - print_stats: Whether to print the anomaly score and changed edge after each perturbation

        Returbs:
        - triple: The poisoned edge_data
        - AS: List of the anomaly score after each perturbation according to surrogate model
        - AS_DOM: List of anomaly score after each perturbation according to DOMINANT
        - AUC_DOM: AUC value after each perturbation according to DOMINANT
        - ACC_DOM: Accuracy of the predicting the target nodes
        - perturb: List of the changed edges
        - edge_index: The edge_index of the poisoned graph, converted to torch format
    """
    triple_copy = triple.copy()
    # print(f'triple copy type: {type(triple_copy)}')
    triple_torch = Variable(torch.from_numpy(triple_copy), requires_grad = True) 
    AS = []
    AS_DOM = []
    AUC_DOM = []
    ACC_DOM = []
    CHANGE_IN_AS_TARGET_NODE_AS = []
    perturb = []
    print("True AS")
    print(model.true_AS(triple_torch).data.detach().cpu().numpy()[0])
    print(model.true_AS(triple_torch).data.detach().cpu().numpy())
    print("After AS")
    AS.append(model.true_AS(triple_torch).data.detach().cpu().numpy()[0])
    AS_DOM_temp, AUC_DOM_temp, ACC_DOM_temp, target_nodes_as = get_DOMINANT_eval_values(DOMINANT_model, config, target_list, perturb)
    CHANGE_IN_AS_TARGET_NODE_AS.append(target_nodes_as)
    AS_DOM.append(AS_DOM_temp)
    AUC_DOM.append(AUC_DOM_temp)
    ACC_DOM.append(ACC_DOM_temp)
    if(print_stats): print('initial anomaly score:', model.true_AS(triple_torch).data.detach().cpu().numpy()[0])
    
    i = 0
    count = 0
    for i in range(1, B+1):   #While we have not reached the maximum number of perturbations
        count += 1
        print("Perturbation number:", count)
        
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
        #slice = [perturb[i][0:2] for i in range(len(perturb))]
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
        #triple_torch = Variable(torch.from_numpy(triple_copy), requires_grad = True)

        # Add perturb to list of perturbs
        perturb.append([int(target_grad[0]),int(target_grad[1]), int(0 < target_grad[2])]) 

        # Get and save updated scores and values
        true_AScore = model.true_AS(triple_torch).data.detach().cpu().numpy()[0] 
        AS.append(true_AScore)
        AS_DOM_temp, AUC_DOM_temp, ACC_DOM_temp, target_node_as = get_DOMINANT_eval_values(DOMINANT_model, config, target_list, perturb)

        CHANGE_IN_AS_TARGET_NODE_AS.append(target_node_as)
        AS_DOM.append(AS_DOM_temp)
        AUC_DOM.append(AUC_DOM_temp)
        ACC_DOM.append(ACC_DOM_temp)
        if(print_stats): print('Iteration:', i, '--- Anomaly score:', true_AScore, '--- DOM anomaly score:', AS_DOM_temp, 
                                '--- DOM AUC:', AUC_DOM_temp, '--- TARGET DOM ACC:', ACC_DOM_temp)

    AS = np.array(AS)    

    edge_index = update_adj_matrix_with_perturb(DOMINANT_model.edge_index, perturb)

    return triple_torch, AS, AS_DOM, AUC_DOM, ACC_DOM, perturb, edge_index, CHANGE_IN_AS_TARGET_NODE_AS

def poison_attack(model, triple, B, print_stats = True):
    triple_copy = triple.copy()
    # print(f'triple copy type: {type(triple_copy)}')
    triple_torch = Variable(torch.from_numpy(triple_copy), requires_grad = True) 
    AS = []
    perturb = []
    inital_AS = model.true_AS(triple_torch).data.numpy()[0]
    AS.append(inital_AS)

    if(print_stats): print('initial anomaly score:', inital_AS)
    
    for i in range(1,B+1):
        loss = model.forward(triple_torch)
        loss.backward()
        
        tmp = triple_torch.grad.data.numpy() # Get gradient of tensor with respect to data, stored in tmp

        grad = np.concatenate((triple_torch[:,:2].data.numpy(),tmp[:,2:]),1) # Concat edge descriptor with gradients
        
        v_grad = np.zeros((len(grad),3)) # 2D array filled w/ zeros

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
        slice = [perturb[i][0:2] for i in range(len(perturb))]
        while v_grad[K][:2].astype('int').tolist() in slice:
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
        if(print_stats): print('Iteration:', i, '--- Anomaly score:', true_AScore)

    AS = np.array(AS)    

    sparse_tensor = triple_torch.to_sparse()

    return triple_torch, AS, perturb

from pygod.detector import DOMINANT
from gad_adversarial_robustness.utils.graph_utils import prepare_graph, adj_matrix_sparse_coo_to_dense
from gad_adversarial_robustness.utils.experiment_results import Experiment
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
    def __init__(self, data, budget = 35, target_node_indexes= []):
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

    adj_adversary, AS, list_of_perturbs = poison_attack(model, triple, budget, print_stats = print_stats)

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

    anomaly_list = np.where(y_binary == True)[0]  # Used for list for which nodes to hide
    print("anomaly_list")

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

    print("Making model...")
    model = multiple_AS(target_lst = anomaly_list, n_node = amount_of_nodes, device = 'cpu')
    budget = 2  # The amount of edges to change

    print("Starting attack...")

    adj_adversary, _, _ = poison_attack(model, triple, budget)

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


    # Testing only for the targetted nodes
    for anomalyIndex in anomaly_list:
        print("auc before poison:")
        print(roc_auc_score(y_binary, score[:, anomalyIndex]))
        print("auc after poison:")
        print(roc_auc_score(y_binary, score_after[:, anomalyIndex]))

    
    return experiment_before_poison, experiment_after_poison #, node_idxs, y_binary








if __name__ == "__main__":
    run_greedy()





