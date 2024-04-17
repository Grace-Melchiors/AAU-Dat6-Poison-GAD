# %% 
from gad_adversarial_robustness.poison.greedy import greedy_attack_with_statistics
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.optim.sgd import SGD
from torch.optim.optimizer import required
from torch.optim import Optimizer
import torch
import sklearn
import numpy as np
import scipy.sparse as sp
from pygod.detector import DOMINANT
from gad_adversarial_robustness.utils.graph_utils import prepare_graph, adj_matrix_sparse_coo_to_dense
from gad_adversarial_robustness.utils.experiment_results import Experiment
import torch
from pygod.utils import load_data
import copy
from typing import Tuple, List, Any
import yaml
from gad_adversarial_robustness.gad.dominant.dominant import Dominant 
from gad_adversarial_robustness.utils.graph_utils import load_anomaly_detection_dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_torch_sparse_tensor
from gad_adversarial_robustness.poison.greedy import multiple_AS
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)

USE_DOMINANT_AS_TO_SELECT_TOP_K_TARGET = True
TOP_K = 20

data = load_data("inj_cora")
y_binary: List[int] = data.y.bool()
print(y_binary)

anomaly_list = np.where(y_binary == True)[0]  # Used for list for which nodes to hide
print(anomaly_list)


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



# %%
script_dir = os.path.abspath('')
yaml_path = os.path.join(script_dir, '..',  'configs', 'dominant_config.yaml')
with open(yaml_path) as file:
        config = yaml.safe_load(file)

dataset_caching_path = os.path.join(script_dir, '..', '..', '..', 'data')


if torch.cuda.is_available():
    config['model']['device'] = 'cuda'
else:
    config['model']['device'] = 'cpu'


clean_data: Data = load_data("inj_cora", dataset_caching_path)

dataset: Data = load_data("inj_cora")
adj, _, _, adj_label = load_anomaly_detection_dataset(clean_data, config['model']['device'])
adj_label = torch.FloatTensor(adj_label).to(config['model']['device'])
edge_index = dataset.edge_index.to(config['model']['device'])
label = torch.Tensor(dataset.y.bool()).to(config['model']['device'])
attrs = dataset.x.to(config['model']['device'])
sparse_adj = to_torch_sparse_tensor(edge_index)


print("Before poison:")
testingModel = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                device=config['model']['device'], edge_index=sparse_adj, adj_label=adj_label, attrs=attrs, label=label)
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

budget = target_list.shape[0] * 2  # The amount of edges to change

budget = 10

print("Starting attack...")
"""
# -------------
dom_model = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                device=config['model']['device'], edge_index=sparse_adj, adj_label=adj_label, attrs=attrs, label=label)

_, AS, AS_DOM, AUC_DOM, ACC_DOM, perturb, edge_index = greedy_attack_with_statistics(
    model, triple, dom_model, config, target_list, budget, print_stats = True)
# -------------
"""

from gad_adversarial_robustness.gad.dominant.dominant_aggr import Dominant as Dominant2
dom_model = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                device=config['model']['device'], edge_index=sparse_adj, adj_label=adj_label, attrs=attrs, label=label)

_, AS2, AS_DOM2, AUC_DOM2, ACC_DOM2, perturb, edge_index = greedy_attack_with_statistics(
    model, triple, dom_model, config, target_list, budget, print_stats = True)
# -------------


# %%
import matplotlib.pyplot as plt

def plot_scores(scores, title='AUC Scores by Budget', xlabel='Budget', ylabel='AUC Score'):
    """
    Plots a list of scores against their corresponding budgets.

    Parameters:
    - scores: List of scores to plot.
    - title: Title of the plot.
    - xlabel: Label for the X-axis.
    - ylabel: Label for the Y-axis.
    """
    budgets = range(1, len(scores) + 1)

    # Creating the plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(budgets, scores, marker='o', linestyle='-', color='b')  # Plotting the scores

    # Adding some flair to the plot
    plt.title(title)  # Title of the plot
    plt.xlabel(xlabel)  # X-axis label
    plt.ylabel(ylabel)  # Y-axis label
    plt.grid(True)  # Adding a grid for better readability

    # Set integer ticks on the X-axis
    plt.xticks(budgets)  # Set integer ticks

    # Display the plot
    plt.show()


plot_scores(AUC_DOM, "DOMINANT AUC-Score by Budget")
plot_scores(AS_DOM, "Sum of Target Nodes Anomaly Scores by Budget", "Budget", "Anomaly Score")
print(AS_DOM)


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
