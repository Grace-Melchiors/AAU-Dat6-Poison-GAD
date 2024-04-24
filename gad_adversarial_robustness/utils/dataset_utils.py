from pygod.generator import gen_contextual_outlier, gen_structural_outlier
from torch_geometric.datasets import AttributedGraphDataset
import torch
import numpy as np
from typing import Tuple, List, Any

def load_dataset_with_anomalies(string, percentage_to_inject = 0.05, nodes_per_clique = 10, seed = 123):
    """
        Loads the dataset based on string "Cora", "Wiki", or "Facebook"
        Injects a percentage of all nodes, default 5%
        Injects a number of nodes per clique, default 10
        seed is used for reproducibility, default 123

        Returns:
            data: the injected data
            anomaly_list: the list of anomalies
    """
    data = load_dataset(string)
    
    amount_of_nodes = data.x.shape[0]

    num_nodes_to_inject = round(amount_of_nodes * percentage_to_inject)
    num_nodes_per_clique = nodes_per_clique
    num_cliques = round((num_nodes_to_inject / 2) / num_nodes_per_clique)
    num_contextual_outliers = num_nodes_to_inject - num_cliques * num_nodes_per_clique

    data, ya = gen_contextual_outlier(data, n = num_contextual_outliers, k = 50, seed = seed) 
    #n (int) – Number of nodes converting to outliers.
    #k (int) – Number of candidate nodes for each outlier node.

    data, ys = gen_structural_outlier(data, m = num_nodes_per_clique, n = num_cliques, seed = seed)
    #m (int) - Number nodes in the outlier cliques.
    #n (int) - Number of outlier clique
    data.y = torch.logical_or(ys, ya).long()

    y_binary: List[int] = data.y.bool()
    anomaly_list = np.where(y_binary == True)[0]  # Used for list for which nodes to hide

    return data, anomaly_list

    
def load_dataset(string):
    """
        Loads the dataset based on string "Cora", "Wiki", or "Facebook"
    """
    dataset = AttributedGraphDataset(root = "data/"+string, name = string)
    data = dataset[0]
    return data
    
