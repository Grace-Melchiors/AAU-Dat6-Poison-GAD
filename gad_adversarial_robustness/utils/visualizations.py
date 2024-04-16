from matplotlib import pyplot as plt
import networkx as nx
import torch
import numpy as np
from torch_geometric.utils import to_networkx
from gad_adversarial_robustness.utils.experiment_results import Experiment
from gad_adversarial_robustness.utils.subgraphs import get_subset_neighbors
from typing import List
from torch_geometric.utils.num_nodes import maybe_num_nodes


def get_rest_of_node_idxs(subset_idxs, num_nodes):
    """
        Function to get the indices of the remaining nodes after removing the nodes specified in subset_idxs.

        - subset_idxs (tensor): The idxs of the nodes not to include.
        - num_nodes (int): The total number of nodes in the graph.

        Returns:
        A tensor containing all node idxs not in the subset
        """

    rest_of_node_idxs = torch.tensor(np.arange(num_nodes))

    for i in subset_idxs:
        rest_of_node_idxs = torch.cat([rest_of_node_idxs[0:i], rest_of_node_idxs[i+1:]])
    
    return rest_of_node_idxs

def visualize_node_degree(edge_index_list, graph_names = None, cut_off = None):
    """
        Visualizes the cumulative distribution of node degrees for a subset for a list of edge_indexes
        Parameters:
        - edge_index_list: List of edge indexes of the graphs
        - graph_names (list of str): The names of the graphs
        - cut_off (int): Limiter for amount of neighbors to display. Default none.

        Returns:
        No return value.
        """
    
    fig = plt.figure(figsize=(9, 4), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)

    # For each subset....
    for i, edge_index in enumerate(edge_index_list):

        # Get amount of neighbors for each node
        num_nodes = maybe_num_nodes(edge_index)
        node_idxs = torch.tensor(np.arange(num_nodes))
        node_idxs_neighbors_amount = []

        # Find degree for every node
        for idx in node_idxs:
            amount = get_subset_neighbors(idx, edge_index).size()[0]
            node_idxs_neighbors_amount.append(amount)

        # Name graph
        graph_name = 'graph' + str(i)
        if(graph_names is not None):
            graph_name = graph_names[i]
        ax.ecdf(node_idxs_neighbors_amount, label=graph_name)
        
   # Label figure
    fig.suptitle("Cumulative distribution of node degrees")

    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Probability of occurrence")
    ax.label_outer()

    if(cut_off is not None):
        ax.set_xbound(0, cut_off)
            
    plt.show()


def visualize_2hop_node_degree(edge_index_list, graph_names = None, cut_off = None):
    """
        Visualizes the cumulative distribution of 2-hop node degrees for a subset for a list of edge_indexes
        Parameters:
        - edge_index_list: List of edge indexes of the graphs
        - graph_names (list of str): The names of the graphs
        - cut_off (int): Limiter for amount of neighbors to display. Default none.

        Returns:
        No return value.
        """
    
    fig = plt.figure(figsize=(9, 4), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)

    # For each subset....
    for i, edge_index in enumerate(edge_index_list):

        # Get amount of neighbors for each node
        num_nodes = maybe_num_nodes(edge_index)
        node_idxs = torch.tensor(np.arange(num_nodes))
        node_idxs_neighbors_amount = []

        # Find 2hop degree for each node
        for idx in node_idxs:
            neighbor_nodes_1hop = get_subset_neighbors(idx, edge_index)  #1 hop
            neighbor_nodes_2hop = get_subset_neighbors(neighbor_nodes_1hop, edge_index)  #2 hop

            neighbor_nodes = torch.cat((neighbor_nodes_1hop, neighbor_nodes_2hop))  # join 1 and 2
            neighbor_nodes = torch.unique(neighbor_nodes)   # Remove duplicates 
            
            amount = neighbor_nodes.size()[0]   # Get amount
            amount = amount - 1 # Subtract self
            node_idxs_neighbors_amount.append(amount)

        # Name graph
        graph_name = 'graph' + str(i)
        if(graph_names is not None):
            graph_name = graph_names[i]
        ax.ecdf(node_idxs_neighbors_amount, label=graph_name)
        
   # Label figure
    fig.suptitle("Cumulative distribution of 2hop node degrees")

    ax.grid(True)
    ax.legend()
    ax.set_xlabel("2hop Node degree")
    ax.set_ylabel("Probability of occurrence")
    ax.label_outer()

    if(cut_off is not None):
        ax.set_xbound(0, cut_off)
            
    plt.show()

    
def visualize_neighbors_amount(edge_index, subset, num_nodes, cut_off = None, normalize = False):
    """
        Visualizes the amount of neighbors for a subset vs the rest of nodes in stacked bar.
        Parameters:
        - edge_index: The edge index of the graph.
        - subset (tensor): The subset of node idxs to visualize.
        - num_nodes (int): The total number of nodes in the graph.
        - cut_off (int): Limiter for amount of neighbors to display. Default none.
        - normalize (bool): Whether to normalize the amount of neighbors to percentages

        Returns:
        No return value.
        """
    
    # Get amount for subset
    subset_neighbors_amount = []
    for idx in subset:
        amount = get_subset_neighbors(idx, edge_index).size()[0]
        subset_neighbors_amount.append(amount)

    # Get amount for rest
    rest_of_node_idxs = get_rest_of_node_idxs(subset, num_nodes)
    rest_of_node_idxs_neighbors_amount = []

    for idx in rest_of_node_idxs:
        amount = get_subset_neighbors(idx, edge_index).size()[0]
        rest_of_node_idxs_neighbors_amount.append(amount)
    
    # Find amouunt of nodes at different intervals
    max_amount = max(subset_neighbors_amount + rest_of_node_idxs_neighbors_amount)
    interval_count = torch.zeros(2, max_amount+1)
    
    for amount in subset_neighbors_amount:
        interval_count[0][amount] += 1
    for amount in rest_of_node_idxs_neighbors_amount:
        interval_count[1][amount] += 1


    if(normalize):  # Normalize to percentages
        interval_count[0] = interval_count[0] / subset.size()[0]
        interval_count[1] = interval_count[1] / rest_of_node_idxs.size()[0]
    
    count = np.arange(max_amount+1) # Counts up till max

    if(cut_off is not None):    # Cut down
        interval_count = interval_count[:,0:cut_off]
        count = count[0:cut_off]
        max_amount = cut_off

    interval_count = interval_count.numpy()

    # # Plot stacked bar
    plt.bar(count+0.4, interval_count[0], color = 'red', width=0.40)    #Subset
    plt.bar(count, interval_count[1], color = 'blue', width=0.40)    #Rest

    plt.show()

def count_positive_class(anomalies_list: List[int]):
    count = 0
    for i in range(len(anomalies_list)):
        if anomalies_list[i] == 1:
            count += 1
    return count

def visualize_graph(experiment: Experiment, y_binary: List[int], node_idxs = None):
    node_colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'yellow'}

    node_size = []
    if node_idxs is not None:
        for i in range(experiment.pred.numel()):
            if i in node_idxs:
                node_size.append(20)
            else:
                node_size.append(6)


    graph3 = to_networkx(experiment.data, to_undirected=True)
    my_pos = nx.spring_layout(graph3, seed = 100)
    node_colors = []
    for i in range(experiment.pred.numel()):
        if experiment.pred[i].item() == 1 and node_idxs is not None and i in node_idxs: 
            print("found red prob")
            node_colors.append('red')
        elif experiment.pred[i].item() == 1:
            print("found orange prob")
            node_colors.append('orange')
        elif experiment.pred[i].item() == 0 and node_idxs is not None and i in node_idxs:
            print("found yellow prob")
            node_colors.append('magenta')
        else:
            node_colors.append('blue')
        
    # Draws the graph ----
    plt.figure(figsize=(12, 12))
    if node_idxs is not None:
        nx.draw(graph3, pos=my_pos, node_size=node_size, node_color=node_colors, edge_color='gray', with_labels=False)
        text = f'Total # of nodes: {experiment.data.size()[0]}. # of poisoned nodes: {len(node_idxs)}. # of labelled anomalies: {count_positive_class(y_binary)}. # of predicted anomalies: {count_positive_class(experiment.pred)}'
        legend_colors = {'red': 'Pred. outlier (poisoned)', 'orange': 'Pred. outlier (not poisoned)', 'magenta': 'Pred. inlier (poisoned)', 'blue': 'Pred. inlier node (not poisoned)'}
    else:
        nx.draw(graph3, pos=my_pos, node_size=6, node_color=node_colors, edge_color='gray', with_labels=False)
        text = f'Total # of nodes: {experiment.data.size()[0]}. # of labelled anomalies: {count_positive_class(y_binary)}. # of predicted anomalies: {count_positive_class(experiment.pred)}'
        legend_colors = {'orange': 'Pred. outlier (not poisoned)', 'blue': 'Pred. inlier node (not poisoned)'}

    legend_colors = {'red': 'Pred. outlier (poisoned)', 'orange': 'Pred. outlier (not poisoned)', 'magenta': 'Pred. inlier (poisoned)', 'blue': 'Pred. inlier node (not poisoned)'}
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for color, label in legend_colors.items()]
    plt.legend(handles=legend_handles, loc='upper right')

    plt.text(0.5, 0.97, text, horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)


    plt.title('CORA Dataset Graph')
    plt.show()
