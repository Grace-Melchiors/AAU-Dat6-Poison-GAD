# %%
from gad_adversarial_robustness.gad.OddBall_vs_DOMININANT import get_OddBall_AS
from gad_adversarial_robustness.poison.greedy import greedy_attack_inserted_edge_statistics, multiple_AS
#Insert Dataset
from pygod.generator import gen_contextual_outlier, gen_structural_outlier
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.datasets import Planetoid
from typing import List
import copy
import yaml
import os
import torch
import numpy as np
from torch_geometric.datasets import AttributedGraphDataset, Planetoid
import math
from pygod.generator import gen_contextual_outlier, gen_structural_outlier
import torch_geometric.transforms as T
from gad_adversarial_robustness.utils.graph_utils import load_injected_dataset, get_anomaly_indexes
from gad_adversarial_robustness.utils.graph_utils import prepare_graph, adj_matrix_sparse_coo_to_dense
from pygod.utils import load_data


# %%

torch.manual_seed(123)
TOP_K = 6

#data, prior_labels = load_injected_dataset("Facebook")
data = load_data("inj_cora") 
label = torch.Tensor(data.y.bool()).to('cuda')
labels_np = label.cpu().detach().numpy()
y_binary: List[int] = data.y.bool()
anomaly_list = np.where(y_binary == True)[0]  # Used for list for which nodes to hide
amount_of_nodes = data.num_nodes


target_list_as = get_OddBall_AS(data=data, device='cuda')
target_list_as = np.array(target_list_as)
target_list = get_anomaly_indexes(target_list_as, labels_np, TOP_K, method='top', print_scores=True, random_top=True)
print(target_list)


# %%

def convertToTriple(data):
    _, adj, _ = prepare_graph(data) #Get adjacency matrix

    n = data.num_nodes

    # 'triple' is a list that will store the perturbed triples during the poisoning process.
    # Each triple represents an edge modification in the form of (node1, node2, edge_label).

    dense_adj = adj.to_dense()
    
    A = np.array(dense_adj)

    triple = []
    for i in range(n):
        for j in range(i+1,n):
        #for j in range(n)
            triple.append([i,j,A[i,j]])
    triple = np.array(triple)

    return triple


triple = convertToTriple(data)

B = 100


model = multiple_AS(target_lst = target_list, n_node = amount_of_nodes, device = 'cuda')



_, _, edge_AS_prior, edge_AS_post, pos_prior, pos_post, perturb = greedy_attack_inserted_edge_statistics(model, triple, target_list, B, CPI = 1, print_stats = True)

# %%
import pickle
def save_results(filename, *args):
    with open(filename, 'wb') as f:
        pickle.dump(args, f)

#save_results("results_node_stats.pkl", edge_AS_prior, edge_AS_post, pos_prior, pos_post, perturb)


# %%

import pickle
def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


loaded_values = load_results("results_node_stats.pkl")
edge_AS_prior, edge_AS_post, pos_prior, pos_post, perturb = loaded_values


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

# Sample data
np.random.seed(42)  # For reproducibility
anomaly_scores = target_list_as  # Replace with your data for all nodes
initial_anomaly_scores = edge_AS_prior  # Replace with your data for benign nodes before connections
post_connection_anomaly_scores = edge_AS_post

# Convert tensors to numpy arrays and flatten them
initial_anomaly_scores = [tensor.cpu().detach().numpy()[0] for tensor in initial_anomaly_scores]
post_connection_anomaly_scores = [tensor.cpu().detach().numpy()[0] for tensor in post_connection_anomaly_scores]

# Filter out scores >= 3.0
initial_anomaly_scores_filtered = []
post_connection_anomaly_scores_filtered = []
for i, score in enumerate(post_connection_anomaly_scores):
    if score < 3.3735 and initial_anomaly_scores[i] < 0.7840:
        initial_anomaly_scores_filtered.append(initial_anomaly_scores[i])
        post_connection_anomaly_scores_filtered.append(score)

initial_anomaly_scores = initial_anomaly_scores_filtered
post_connection_anomaly_scores = post_connection_anomaly_scores_filtered

initial_anomaly_scores.append(5.1523)
initial_anomaly_scores.append(3.3735)
initial_anomaly_scores.append(3.0791)



post_connection_anomaly_scores.append(4.4713)
post_connection_anomaly_scores.append(4.4713)
post_connection_anomaly_scores.append(4.4713)

# Calculate percentiles with respect to the entire set of anomaly scores
percentiles_initial = [percentileofscore(anomaly_scores, score) for score in initial_anomaly_scores]
percentiles_post = [percentileofscore(anomaly_scores, score) for score in post_connection_anomaly_scores]

# Prepare DataFrame for easier handling
data = pd.DataFrame({
    'initial_score': initial_anomaly_scores,
    'post_connection_score': post_connection_anomaly_scores,
    'percentile_initial': percentiles_initial,
    'percentile_post': percentiles_post
})


from brokenaxes import brokenaxes
# Create broken y-axis
fig = plt.figure(figsize=(12, 8))
bax = brokenaxes(ylims=((-0.25, 1.0), (3, 5.5)), hspace=0.08)

# Plot initial scores
initial_scatter = bax.scatter(data['percentile_initial'], data['initial_score'], color='blue', label='Initial Scores', marker='o', alpha=0.6, s=100)

# Plot post-connection scores
post_scatter = bax.scatter(data['percentile_post'], data['post_connection_score'], color='red', label='Post-Connection Scores', marker='x', alpha=0.6, s=100)

bax.tick_params(axis='both', which='major', labelsize=12)

# Connecting lines
for i in range(len(data)):
    if data['initial_score'][i] <= 1.5 or data['post_connection_score'][i] >= 3:
        bax.plot([data['percentile_initial'][i], data['percentile_post'][i]], 
                 [data['initial_score'][i], data['post_connection_score'][i]], 
                 color='gray', linestyle='--', linewidth=1.0, alpha=0.8)

# Add text annotations for overlapping points
for column, color, scatter in zip(['percentile_initial', 'percentile_post'], ['blue', 'red'], [initial_scatter, post_scatter]):
    counts = data[column].value_counts()
    for index, count in counts.items():
        if count >= 1:
            y_mean = np.mean(data.loc[data[column] == index, 'initial_score' if column == 'percentile_initial' else 'post_connection_score'])
            y_offset = 0.14 if color == 'blue' else -0.14
            bax.text(index,  # x-coordinate
                     y_mean + y_offset,  # y-coordinate for text placement with offset
                     f'{count}',  # Text showing the count of overlapping points
                     ha='center', va='center', fontsize=14, color=color)

# Labels and legend
bax.set_xlabel('Percentile Rank', fontsize=12)
bax.set_ylabel('Anomaly Score', fontsize=12)
fig.suptitle('OddBall Anomaly Scores Before and After Connection by Anomalies', fontsize=18, y=0.93)

# Set legend with larger font size
bax.legend(fontsize=14, loc='upper left')
bax.grid(True)

plt.show()

# %%
