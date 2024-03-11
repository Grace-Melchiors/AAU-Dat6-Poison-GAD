"""
# Structure
A folder for datasets, which includes original datasets, with anomalies, and poisoned version

A folder for anomaly injection

A folder for poisons

A folder for GADs

A folder for results, includes raw data, and graphs

A main file which calls these as needed
"""


from torch_geometric.datasets import Planetoid
import torch_geometric.utils as pyg_utils
import networkx as nx
import matplotlib.pyplot as plt
import torch

import torch_geometric.transforms as T


#Import from different folders with stuff



#Controls, select dataset, anomaly injection, poison, run methods, and display options

DATASET_NAME = 'Cora'  #Options: 'Cora', 


if __name__ == '__main__':
### LoadDataset ###
    # the [0] at the end is used to access the attributes
    data = Planetoid('./data/'+DATASET_NAME, DATASET_NAME, transform=T.NormalizeFeatures())[0]
    dataset_name = 'Cora'

    # Load the dataset
    dataset = Planetoid(root='./data', name=DATASET_NAME)

    # Access the dataset attributes
    data = dataset[0]

    # Convert PyTorch Geometric data to NetworkX graph ---
    base_graph_dataset = pyg_utils.to_networkx(data, to_undirected=True)



### Inject anomalies in dataset ###
    from pygod.generator import gen_contextual_outlier, gen_structural_outlier

    # introduces anomalies (maybe poison I can't tell) to the train data in case the original dataset didnt have any
    data, ya = gen_contextual_outlier(data, n=100, k=50) #In
    data, ys = gen_structural_outlier(data, m=10, n=10)
    data.y = torch.logical_or(ys, ya).long()

    # Convert PyTorch Geometric data to NetworkX graph ---
    clean_graph_dataset = pyg_utils.to_networkx(data, to_undirected=True)


### Poison dataset ###


### Run model(s) ###


### Compare clean model with poison or/and get performance at different amounts of poison ###



### Produce results ###











