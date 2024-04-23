from gad_adversarial_robustness.poison.greedy import multiple_AS
from gad_adversarial_robustness.utils.graph_utils import prepare_graph
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


def compare_OddBall_DOMINANT(data, DOMINANT_model, config, k, with_plot = False, print_accuracy = True):
    """
        Parameters:
            data: graph data
            DOMINANT_model: DOMINANT model
            config: configrations to run model
            k: top k nodes of OddBall and DOMINANT AS scores
            with_plot: whether to plot graphs
            print_accuracy: whether to print accuracy

        Returns:
            topK_dom: top k nodes of DOMINANT AS scores
            topK_odd: top k nodes of OddBall AS scores
    """

    # Get anomaly scores
    DOMINANT_AS = get_DOMINANT_AS(DOMINANT_model, config)
    OddBall_AS = get_OddBall_AS(data, config['model']['device'])

    # Get sorted anomaly scores by index
    sorted_dom_as = np.argsort(DOMINANT_AS)[::-1]
    sorted_odd_as = np.argsort(OddBall_AS)[::-1]

    # Get top k
    topK_dom = sorted_dom_as[:k]
    topK_odd = sorted_odd_as[:k]

    # Get list of anomalies
    anomaly_list = np.where(DOMINANT_model.label == True)[0]  # Used for list for which nodes are anomalies

    # Print accuracy
    if print_accuracy:
        print("Dominant's anomalies within top "+ str(k) +" nodes:",len(set(topK_dom).intersection(set(anomaly_list))))
        print("OddBall's anomalies within top "+ str(k) +" nodes:",len(set(topK_odd).intersection(set(anomaly_list))))

    # Plot
    if with_plot:
        # get list of nodes that are in atleast one of the lists
        topK_all = list(set(topK_dom).union(set(topK_odd)))

        indexes_dom = []
        indexes_odd = []
        indexes_dom_anom = []
        indexes_odd_anom = []

        # Append their spot in each list, and give them color
        for i in topK_all:
            if i in anomaly_list:
                indexes_dom_anom.append(np.where(sorted_dom_as == i)[0][0])
                indexes_odd_anom.append(np.where(sorted_odd_as == i)[0][0])
            else :
                indexes_dom.append(np.where(sorted_dom_as == i)[0][0])
                indexes_odd.append(np.where(sorted_odd_as == i)[0][0])

        plt.plot(indexes_dom, indexes_odd, 'o', color = 'blue')
        plt.plot(indexes_dom_anom, indexes_odd_anom, 'o', color = 'red')
        plt.xlabel('DOMINANT AS_Rank')
        plt.ylabel('OddBall AS_Rank')
        plt.show()
    
    return topK_dom, topK_odd

def get_DOMINANT_AS(DOMINANT_model, config):
    
    DOMINANT_model.to(config['model']['device'])
    DOMINANT_model.fit(config, verbose=False)

    return DOMINANT_model.score

class OddBall_AS(nn.Module):
    def __init__(self, n_node, device):
        """
            target_lst (numpy.ndarray): The target list to be initialized.
            n_node (int): The number of nodes.  
            device (str): The device to be used.
        """
        super().__init__()
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
        N = torch.sum(A, 1).to(self.device)
        E = torch.sum(A, 1).to(self.device) + 0.5 * torch.diag(self.sparse_matrix_power(A, 3)).T.to(self.device)
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

        AS_list = []    

        for i in range(self.n):

            tmp = (torch.max(E[i],torch.exp(b)*(N[i]**w)).to(self.device)\
                   /torch.min(E[i],torch.exp(b)*(N[i]**w)).to(self.device))*\
                    torch.log(torch.abs(E[i]-torch.exp(b)*(N[i]**w))+1).to(self.device)
            AS_list.append(tmp.item())
        return AS_list 


def get_OddBall_AS(data, device, OddBall_AS = OddBall_AS):
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

    model = OddBall_AS(n_node = amount_of_nodes, device = device)

    OddBall_AS = model.true_AS(triple)

    return OddBall_AS
