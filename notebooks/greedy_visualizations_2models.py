# %% 
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse, to_undirected, to_dense_adj
from gad_adversarial_robustness.utils.graph_utils import top_anomalous_nodes, load_injected_dataset, get_anomaly_indexes, get_anomalies_with_label_1
from gad_adversarial_robustness.poison.greedy import greedy_attack_with_statistics, greedy_attack_with_statistics_multi
import os
import torch
import numpy as np
from torch.optim.optimizer import required
import torch
import numpy as np
import scipy.sparse as sp
#from pygod.detector import DOMINANT
from gad_adversarial_robustness.utils.graph_utils import prepare_graph, adj_matrix_sparse_coo_to_dense
import torch
from pygod.utils import load_data
from typing import Tuple, List, Any
import yaml
import os
from gad_adversarial_robustness.gad.dominant.dominant_cuda_v2 import Dominant 
from gad_adversarial_robustness.gad.dominant.dominant_cuda_medoid import Dominant as DominantAgg
from gad_adversarial_robustness.gad.dominant.dominant_cuda_preprocess import Dominant as DominantPP
from gad_adversarial_robustness.gad.dominant.dominant_cuda_preprocess_and_medoid import Dominant as DominantAggPP
from gad_adversarial_robustness.gad.dominant.dominant_cuda_preprocess_ob import Dominant as DominantNew
from gad_adversarial_robustness.gad.dominant.dominant_cuda_preprocess_ob_v2 import Dominant as DominantNew2
from gad_adversarial_robustness.utils.graph_utils import load_anomaly_detection_dataset
from torch_geometric.data import Data
from gad_adversarial_robustness.poison.greedy import multiple_AS
from gad_adversarial_robustness.utils.graph_utils import get_n_anomaly_indexes


from gad_adversarial_robustness.gad.OddBall_vs_DOMININANT import get_OddBall_AS
from pygod.detector import DOMINANT

torch.manual_seed(123)

USE_DOMINANT_AS_TO_SELECT_TOP_K_TARGET = False
USE_ODDBALL_AS_TO_SELECT_TOP_K_TARGET = True
USE_FIRST_K_TO_SELECT_TOP_K_TARGET = False

TOP_K = 15
SAMPLE_RANDOM_TOP = True

SAMPLE_MODE = 'top' # 'top', 'lowest', 'normal'
DATASET_NAME = 'inj_cora'

print(DATASET_NAME)
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

prior_labels = None

if DATASET_NAME == 'inj_cora' or DATASET_NAME == 'inj_amazon':
    dataset: Data = load_data(DATASET_NAME)
    dataset_planetoid = Planetoid(root='data', name='Cora')
    prior_labels = dataset_planetoid[0].y

elif DATASET_NAME == 'Wiki' or DATASET_NAME == 'Cora' or DATASET_NAME == 'Facebook':
    dataset, prior_labels = load_injected_dataset(DATASET_NAME)

if GRAPH_PARTITION_SIZE is not None:
    subset = torch.randperm(dataset.num_nodes)[:GRAPH_PARTITION_SIZE]
    dataset = dataset.subgraph(subset)

adj, _, _, adj_label = load_anomaly_detection_dataset(dataset, config['model']['device'])
adj_label = torch.FloatTensor(adj_label).to(config['model']['device'])
#edge_index = dense_to_sparse(torch.tensor(adj))[0].to(config['model']['device'])
edge_index = to_undirected(dataset.edge_index).to(config['model']['device'])
label = torch.Tensor(dataset.y.bool()).to(config['model']['device'])
attrs = dataset.x.to(config['model']['device'])
#sparse_adj = to_torch_sparse_tensor(edge_index)

y_binary: List[int] = dataset.y.bool()

_, adj, _ = prepare_graph(dataset)

dense_adj = to_dense_adj(edge_index)[0].detach().cpu().numpy()

amount_of_nodes = dataset.num_nodes

# 'triple' is a list that will store the perturbed triples during the poisoning process.
# Each triple represents an edge modification in the form of (node1, node2, edge_label).

#dense_adj = adj.to_dense()  #Fill in zeroes where there are no edges
#A = dense_adj.detach().cpu().numpy()


print("Create poison compatible adjacency matrix...")
triple = []
for i in range(amount_of_nodes):
    for j in range(i + 1, amount_of_nodes):
        triple.append([i, j, dense_adj[i,j]])  #Fill with 0, then insert actual after

print(type(triple))
triple = np.array(triple)



# %%

print("Before poison:")
testingModel = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                device=config['model']['device'], edge_index=edge_index, adj_label=adj_label, attrs=attrs, label=label, prior_labels=prior_labels)
testingModel.to(config['model']['device'])
testingModel.fit(config, new_edge_index=edge_index, attrs=attrs, verbose=False, top_k=TOP_K)

target_list = []
if USE_DOMINANT_AS_TO_SELECT_TOP_K_TARGET:
    print("TOP K_AS:")
    print(testingModel.top_k_AS) 
    target_list = np.array(testingModel.top_k_AS)
elif USE_ODDBALL_AS_TO_SELECT_TOP_K_TARGET:
    target_list_as = get_OddBall_AS(data=dataset, device=config['model']['device'])
    target_list_as = np.array(target_list_as)
    labels_np = label.cpu().detach().numpy()
    print("ALL:")
    get_anomalies_with_label_1(target_list_as, labels_np)
    print("NOT ALL:")
    if SAMPLE_RANDOM_TOP:
        target_list = get_anomaly_indexes(target_list_as, labels_np, TOP_K, method=SAMPLE_MODE, print_scores=True, random_top=True)
    else: 
        target_list = get_anomaly_indexes(target_list_as, labels_np, TOP_K, method=SAMPLE_MODE, print_scores=True, random_top=False)
    print(f'Target list: {target_list}')
elif USE_FIRST_K_TO_SELECT_TOP_K_TARGET:
    anomaly_list = get_n_anomaly_indexes(y_binary, TOP_K)
    target_list = anomaly_list

print("Making model...")
model = multiple_AS(target_lst = target_list, n_node = amount_of_nodes, device = config['model']['device'])

for target in target_list:
    print(f'ID: {target}, Target: {label[target]}')

#budget = target_list.shape[0] * 2  # The amount of edges to change

budget = TOP_K * 6

print("Starting attack...")
"""
# -------------
dom_model = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                device=config['model']['device'], edge_index=sparse_adj, adj_label=adj_label, attrs=attrs, label=label)

_, AS, AS_DOM, AUC_DOM, ACC_DOM, perturb, edge_index = greedy_attack_with_statistics(
    model, triple, dom_model, config, target_list, budget, print_stats = True)
# -------------
"""

dom_params = {'feat_size': attrs.size(1), 'hidden_size': config['model']['hidden_dim'], 'dropout': config['model']['dropout'],
                'device': config['model']['device'], 'edge_index': edge_index, 'adj_label': adj_label, 'attrs': attrs, 'label': label, 'prior_labels': prior_labels}


#_, AS, AS_DOM, AUC_DOM, ACC_DOM, perturb, edge_index = greedy_attack_with_statistics(
#    model, triple, dom_model, config, target_list, budget, print_stats = True)

#dom_model_1 = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
#                device=config['model']['device'], edge_index=edge_index, adj_label=adj_label, attrs=attrs, label=label)

#_, AS_1, AS_DOM_1, AUC_DOM_1, ACC_DOM_1, perturb_1, edge_index_1, CHANGE_IN_TARGET_NODE_AS_1 = greedy_attack_with_statistics(
#    model, triple, dom_model_1, config, target_list, budget, print_stats = True)

## MODIFIED
#dom_model_2 = DominantAgg(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
#                device=config['model']['device'], edge_index=edge_index, adj_label=adj_label, attrs=attrs, label=label)

#_, AS_2, AS_DOM_2, AUC_DOM_2, ACC_DOM_2, perturb_2, edge_index_2, CHANGE_IN_TARGET_NODE_AS_2 = greedy_attack_with_statistics(
#    model, triple, dom_model_2, config, target_list, budget, print_stats = True)

#normal_dominant = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
#                     device=config['model']['device'], edge_index=edge_index, adj_label=adj_label, attrs=attrs, label=label)


_, AS_1, AS_DOM_1, AUC_DOM_1, ACC_DOM_1, perturb_1, edge_index_1, CHANGE_IN_TARGET_NODE_AS_1, LAST_FEAT_LOSS, LAST_STRUCT_LOSS = greedy_attack_with_statistics_multi(
    model, triple, Dominant, dom_params, config, target_list, budget, print_stats = True, DOMINANT_model_2=DominantNew2)
#_, AS_1, AS_DOM_1, AUC_DOM_1, ACC_DOM_1, perturb_1, edge_index_1, CHANGE_IN_TARGET_NODE_AS_1, LAST_FEAT_LOSS, LAST_STRUCT_LOSS = greedy_attack_with_statistics_multi(
#    model, triple, Dominant, dom_params, config, target_list, budget, print_stats = True, DOMINANT_model_2=DominantNew, DOMINANT_model_3=DominantAgg, DOMINANT_model_4=DominantPP)

# %%
torch.save(edge_index_1, 'edge_index_10_50.pt')

# -------------


# %%
import matplotlib.pyplot as plt



def plot_scores(scores1, scores2, title='AUC Scores by Budget', xlabel='Budget', ylabel='AUC Score', scores3=None, scores4=None):
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
    plt.plot(budgets, scores2, marker='o', linestyle='-', color='r', label='Our proposal v1')  # Plotting the second set of scores
    if scores3 is not None:
        plt.plot(budgets, scores3, marker='o', linestyle='-', color='g', label='Our proposal v2')  # Plotting the third set of scores
    if scores4 is not None:
        plt.plot(budgets, scores4, marker='o', linestyle='-', color='y', label='DOMINANT w/ Jaccard & medoid')  # Plotting the third set of scores

    # Adding some flair to the plot
    plt.title(title)  # Title of the plot
    plt.xlabel(xlabel)  # X-axis label
    plt.ylabel(ylabel)  # Y-axis label
    plt.grid(True)  # Adding a grid for better readability

    # Set integer ticks on the X-axis
    plt.xticks(budgets)  # Set integer ticks

    plt.legend()

    # Display the plot
    plt.show()





#plot_scores(AUC_DOM2, "DOMINANT AUC-Score by Budget")
#plot_scores(AS_DOM2, "Sum of Target Nodes Anomaly Scores by Budget", "Budget", "Anomaly Score")
#print(AS_DOM2)

def plot_score_percentage_change(scores1, scores2, title='Percentage Change of Scores', xlabel='Budget', ylabel='Percentage Change', scores3=None, scores4=None):
    """
    Plots the percentage change of scores compared to the first set of scores.

    Parameters:
    - scores1: List of scores for the baseline.
    - scores2: List of scores to compare against the baseline.
    - title: Title of the plot.
    - xlabel: Label for the X-axis.
    - ylabel: Label for the Y-axis.
    - scores3: Optional third set of scores to compare against the baseline.
    """
    def normalize_scores(scores):
        max_score = max(scores)
        return [score / max_score for score in scores]

    def calculate_percentage_changes(scores, baseline):
        return [(score - baseline) / baseline * 100 for score in scores]

    normalized_scores1 = normalize_scores(scores1)
    normalized_scores2 = normalize_scores(scores2)
    if scores3 is not None:
        normalized_scores3 = normalize_scores(scores3)
    if scores4 is not None:
        normalized_scores4 = normalize_scores(scores4)

    percentage_changes1 = calculate_percentage_changes(normalized_scores1, normalized_scores1[0])
    percentage_changes2 = calculate_percentage_changes(normalized_scores2, normalized_scores1[0])
    if scores3 is not None:
        percentage_changes3 = calculate_percentage_changes(normalized_scores3, normalized_scores1[0])
    if scores4 is not None:
        percentage_changes4 = calculate_percentage_changes(normalized_scores4, normalized_scores1[0])

    budgets = range(0, len(scores1))

    # Creating the plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(budgets, percentage_changes1, marker='o', linestyle='-', color='b', label='Unmodified DOMINANT')
    plt.plot(budgets, percentage_changes2, marker='o', linestyle='-', color='r', label='Our proposal')
    if scores3 is not None:
        plt.plot(budgets, percentage_changes3, marker='o', linestyle='-', color='g', label='DOMINANT w/ Soft Medoid')
    if scores4 is not None:
        plt.plot(budgets, percentage_changes4, marker='o', linestyle='-', color='y', label='DOMINANT w/ Jaccard')

    # Adding some flair to the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # Set integer ticks on the X-axis
    plt.xticks(budgets)

    plt.legend()

    # Display the plot
    plt.show()

# %%



def print_values_not_in_bigger_array(small_array, bigger_array):
    values_not_in_bigger_array = [value for value in small_array if value not in bigger_array]
    if values_not_in_bigger_array:
        print("Indexes of nodes selected by topK AS by DOMINANT, that are not present in the list of injected anomalies:", values_not_in_bigger_array)
    else:
        print("All values in the small array are present in the bigger array.")

#target_list = np.array(testingModel.top_k_AS)
#print_values_not_in_bigger_array(target_list, anomaly_list)


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


def plot_anomaly_scores(anomaly_scores, model_name):
    iterations = range(len(anomaly_scores))  # Number of iterations

    for node_index in range(len(anomaly_scores[0])):
        node_scores = [scores[node_index] for scores in anomaly_scores]
        plt.plot(iterations, node_scores)

    plt.xlabel('Budget')
    plt.ylabel('Anomaly Score')
    plt.title(f'Anomaly Score Development for Each Node: {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_anomaly_scores(CHANGE_IN_TARGET_NODE_AS_1[0], "Unmodified DOMINANT")
plot_anomaly_scores(CHANGE_IN_TARGET_NODE_AS_1[1], "Our proposal")
#plot_anomaly_scores(CHANGE_IN_TARGET_NODE_AS_1[2], "DOMINANT Jaccard")

# %%
plot_scores(AS_DOM_1[0], AS_DOM_1[1], "Sum of Target Nodes Anomaly Scores by Budget", "Budget", "Anomaly Score")
plot_scores(AUC_DOM_1[0], AUC_DOM_1[1], "AUC by Budget", "Budget", "Anomaly Score")

#print(AS_DOM2)

#AUC_DOM3 = [0.8164340495122089, 0.8177578525912143, 0.8156445609879884, 0.816068911069757, 0.8151440816556703, 0.8127586984717758, 0.8129983646309142, 0.8126825692212261, 0.8127192240455648, 0.8119692099475553, 0.8119748491512998, 0.8100039474426212, 0.8085321152653245, 0.8071307731348334, 0.8061157164608357, 0.80334686742232, 0.8060818812383691, 0.8045423786161394, 0.7949275362318842, 0.8053600631590819, 0.805606778322901, 0.8105241639880449, 0.8179101110923138, 0.8117098065753114, 0.8077835109682512, 0.8026758021767327, 0.8088958439068404, 0.7987678339818417, 0.8021358484182033, 0.7943565668527603, 0.7997730220492867]
#AS_DOM3 = [42.31573, 42.33776, 42.281334, 42.30277, 42.298187, 42.23702, 42.300823, 42.258064, 42.257446, 42.23358, 42.220634, 42.19507, 42.051605, 41.77787, 41.589165, 41.44771, 41.76278, 41.445587, 41.004654, 41.27797, 40.98237, 41.213097, 41.853844, 40.96518, 40.93633, 40.30526, 40.568085, 39.9445, 40.14162, 39.60869, 39.669918]
plot_scores(AUC_DOM_1[0], AUC_DOM_1[1], "AUC-Score by Budget", "Budget", "Anomaly Score", scores3=AUC_DOM_1[2], scores4=AUC_DOM_1[3])
plot_scores(AS_DOM_1[0], AS_DOM_1[1], "Sum of Target Nodes Anomaly Scores by Budget", "Budget", "Anomaly Score", scores3=AS_DOM_1[3])
# %%

# %%

plot_score_percentage_change(AUC_DOM_1[0], AUC_DOM_1[1], "AUC by Budget", "Budget", "Anomaly Score % Change", scores3=AUC_DOM_1[2], scores4=AUC_DOM_1[3])
plot_score_percentage_change(AS_DOM_1[0], AS_DOM_1[1], "Sum of Target Nodes Anomaly Scores by Budget", "Budget", "Anomaly Score % Change", scores3=AS_DOM_1[2], scores4=AS_DOM_1[3])
plot_score_percentage_change(LAST_STRUCT_LOSS[0], LAST_STRUCT_LOSS[1], "Last Epoch Structure Loss By Budget", "Budget", "Anomaly Score % Change", scores3=LAST_STRUCT_LOSS[2], scores4=LAST_STRUCT_LOSS[3])
# %%

print(AS_1)
# %%
