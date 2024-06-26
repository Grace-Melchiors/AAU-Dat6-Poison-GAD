# %% 
from gad_adversarial_robustness.poison.greedy import greedy_attack_with_statistics
import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.optim.sgd import SGD
from torch.optim.optimizer import required
import torch
import numpy as np
import scipy.sparse as sp
#from pygod.detector import DOMINANT
from gad_adversarial_robustness.utils.graph_utils import prepare_graph, adj_matrix_sparse_coo_to_dense
from gad_adversarial_robustness.utils.experiment_results import Experiment
import torch
from pygod.utils import load_data
import copy
from typing import Tuple, List, Any
import yaml
from gad_adversarial_robustness.gad.dominant.dominant_cuda import Dominant 
from gad_adversarial_robustness.gad.dominant.dominant_cuda_sage import DominantAgg
from gad_adversarial_robustness.utils.graph_utils import load_anomaly_detection_dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_torch_sparse_tensor
from gad_adversarial_robustness.poison.greedy import multiple_AS
from gad_adversarial_robustness.utils.graph_utils import get_n_anomaly_indexes
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)

USE_DOMINANT_AS_TO_SELECT_TOP_K_TARGET = False
TOP_K = 40
DATASET_NAME = 'inj_cora'
GRAPH_PARTITION_SIZE = None

script_dir = os.path.abspath('')
yaml_path = os.path.join(script_dir, '..',  'configs', 'dominant_config.yaml')
with open(yaml_path) as file:
        config = yaml.safe_load(file)

dataset_caching_path = os.path.join(script_dir, '..', '..', '..', 'data')


if torch.cuda.is_available():
    config['model']['device'] = 'cuda'
else:
    config['model']['device'] = 'cpu'

dataset: Data = load_data(DATASET_NAME)
if GRAPH_PARTITION_SIZE is not None:
    subset = torch.randperm(dataset.num_nodes)[:GRAPH_PARTITION_SIZE]
    dataset = dataset.subgraph(subset)



print("TEST")
print(dataset.num_nodes)


adj, _, _, adj_label = load_anomaly_detection_dataset(dataset, config['model']['device'])
adj_label = torch.FloatTensor(adj_label).to(config['model']['device'])
edge_index = dataset.edge_index.to(config['model']['device'])
label = torch.Tensor(dataset.y.bool()).to(config['model']['device'])
attrs = dataset.x.to(config['model']['device'])
#sparse_adj = to_torch_sparse_tensor(edge_index)

y_binary: List[int] = dataset.y.bool()
print(y_binary)

anomaly_list = get_n_anomaly_indexes(y_binary, TOP_K)
#anomaly_list = np.where(y_binary == True)[0]  # Used for list for which nodes to hide

print(anomaly_list)

_, adj, _ = prepare_graph(dataset)

amount_of_nodes = dataset.num_nodes

# 'triple' is a list that will store the perturbed triples during the poisoning process.
# Each triple represents an edge modification in the form of (node1, node2, edge_label).

dense_adj = adj.to_dense()  #Fill in zeroes where there are no edges


print("Create poison compatible adjacency matrix...")
triple = []
for i in range(amount_of_nodes):
    for j in range(i + 1, amount_of_nodes):
        triple.append([i, j, dense_adj[i,j]])  #Fill with 0, then insert actual after

triple = np.array(triple)



# %%

print("Before poison:")
testingModel = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                device=config['model']['device'], edge_index=edge_index, adj_label=adj_label, attrs=attrs, label=label)
testingModel.to(config['model']['device'])
testingModel.fit(config, verbose=False, top_k=TOP_K)

if USE_DOMINANT_AS_TO_SELECT_TOP_K_TARGET:
    print("TOP K_AS:")
    print(testingModel.top_k_AS) 
    target_list = np.array(testingModel.top_k_AS)
else:
    target_list = anomaly_list


print("Making model...")
model = multiple_AS(target_lst = target_list, n_node = amount_of_nodes, device = config['model']['device'])

#budget = target_list.shape[0] * 2  # The amount of edges to change

budget = 30

print("Starting attack...")
"""
# -------------
dom_model = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                device=config['model']['device'], edge_index=sparse_adj, adj_label=adj_label, attrs=attrs, label=label)

_, AS, AS_DOM, AUC_DOM, ACC_DOM, perturb, edge_index = greedy_attack_with_statistics(
    model, triple, dom_model, config, target_list, budget, print_stats = True)
# -------------
"""

dom_model = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                device=config['model']['device'], edge_index=edge_index, adj_label=adj_label, attrs=attrs, label=label)

_, AS2, AS_DOM2, AUC_DOM2, ACC_DOM2, perturb, edge_index, CHANGE_IN_TARGET_NODE_AS = greedy_attack_with_statistics(
    model, triple, dom_model, config, target_list, budget, print_stats = True)

# -------------


# %%
import matplotlib.pyplot as plt



def plot_scores(scores1, scores2, title='AUC Scores by Budget', xlabel='Budget', ylabel='AUC Score'):
    """
    Plots a list of scores against their corresponding budgets.

    Parameters:
    - scores: List of scores to plot.
    - title: Title of the plot.
    - xlabel: Label for the X-axis.
    - ylabel: Label for the Y-axis.
    """
    budgets = range(0, len(scores1))

    # Creating the plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    #plt.plot(budgets, scores, marker='o', linestyle='-', color='b')  # Plotting the scores
    plt.plot(budgets, scores1, marker='o', linestyle='-', color='b', label='Unmodified DOMINANT')  # Plotting the first set of scores
    plt.plot(budgets, scores2, marker='o', linestyle='-', color='r', label='Modified DOMINANT')  # Plotting the second set of scores

    # Adding some flair to the plot
    plt.title(title)  # Title of the plot
    plt.xlabel(xlabel)  # X-axis label
    plt.ylabel(ylabel)  # Y-axis label
    plt.grid(True)  # Adding a grid for better readability

    # Set integer ticks on the X-axis
    plt.xticks(budgets)  # Set integer ticks

    # Display the plot
    plt.show()


#plot_scores(AUC_DOM2, "DOMINANT AUC-Score by Budget")
#plot_scores(AS_DOM2, "Sum of Target Nodes Anomaly Scores by Budget", "Budget", "Anomaly Score")
print(AS_DOM2)


# %%



def print_values_not_in_bigger_array(small_array, bigger_array):
    values_not_in_bigger_array = [value for value in small_array if value not in bigger_array]
    if values_not_in_bigger_array:
        print("Indexes of nodes selected by topK AS by DOMINANT, that are not present in the list of injected anomalies:", values_not_in_bigger_array)
    else:
        print("All values in the small array are present in the bigger array.")

target_list = np.array(testingModel.top_k_AS)
print_values_not_in_bigger_array(target_list, anomaly_list)


# %%
def plot_anomaly_scores(anomaly_scores):
    iterations = range(1, len(anomaly_scores[0]))

    for node_index, scores in enumerate(anomaly_scores, start=1):
        plt.plot(iterations, scores, label=f'Node {node_index}')

    plt.xlabel('Budget')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Score Development for Each Node')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_anomaly_scores(anomaly_scores):
    iterations = range(len(anomaly_scores))  # Number of iterations

    for node_index in range(len(anomaly_scores[0])):
        node_scores = [scores[node_index] for scores in anomaly_scores]
        plt.plot(iterations, node_scores)

    plt.xlabel('Budget')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Score Development for Each Node')
    plt.legend()
    plt.grid(True)
    plt.show()
plot_anomaly_scores(CHANGE_IN_TARGET_NODE_AS)

# %%

print(AS_DOM2)

AUC_DOM3 = [0.8164340495122089, 0.8177578525912143, 0.8156445609879884, 0.816068911069757, 0.8151440816556703, 0.8127586984717758, 0.8129983646309142, 0.8126825692212261, 0.8127192240455648, 0.8119692099475553, 0.8119748491512998, 0.8100039474426212, 0.8085321152653245, 0.8071307731348334, 0.8061157164608357, 0.80334686742232, 0.8060818812383691, 0.8045423786161394, 0.7949275362318842, 0.8053600631590819, 0.805606778322901, 0.8105241639880449, 0.8179101110923138, 0.8117098065753114, 0.8077835109682512, 0.8026758021767327, 0.8088958439068404, 0.7987678339818417, 0.8021358484182033, 0.7943565668527603, 0.7997730220492867]
AS_DOM3 = [42.31573, 42.33776, 42.281334, 42.30277, 42.298187, 42.23702, 42.300823, 42.258064, 42.257446, 42.23358, 42.220634, 42.19507, 42.051605, 41.77787, 41.589165, 41.44771, 41.76278, 41.445587, 41.004654, 41.27797, 40.98237, 41.213097, 41.853844, 40.96518, 40.93633, 40.30526, 40.568085, 39.9445, 40.14162, 39.60869, 39.669918]
plot_scores(AUC_DOM2, AUC_DOM3, "Sum of Target Nodes Anomaly Scores by Budget", "Budget", "Anomaly Score")
plot_scores(AS_DOM2, AS_DOM3, "Sum of Target Nodes Anomaly Scores by Budget", "Budget", "Anomaly Score")
# %%
