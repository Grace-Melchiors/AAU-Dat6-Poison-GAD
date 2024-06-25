# %%
from pygod.utils import load_data
import seaborn as sns


"""

1. Get the graph (inj_cora)
2. Get the average Jaccard similarity of all connections of nodes with labels = 1
3. Get the average Jaccard similarity of all connections of nodes with labels = 0
4. Boxplot?

"""

import torch
from torch_geometric.utils import to_undirected, subgraph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def get_neighbours(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True
    adj = adj | adj.t()  # Make the graph undirected
    neighbours = [set(adj[i].nonzero(as_tuple=True)[0].tolist()) for i in range(num_nodes)]
    return neighbours

def _jaccard_similarity(a, b):
    intersection = np.count_nonzero(a.multiply(b))
    J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)
    return J

def analyze_neighbours(data):
    edge_index = to_undirected(data.edge_index)
    num_nodes = data.num_nodes
    neighbours = get_neighbours(edge_index, num_nodes)
    
    indexes_y0 = (data.y == 0).nonzero(as_tuple=True)[0]
    indexes_y1 = (data.y == 1).nonzero(as_tuple=True)[0]
    indexes_y2 = (data.y == 2).nonzero(as_tuple=True)[0]
    indexes_y3 = (data.y == 3).nonzero(as_tuple=True)[0]
    
    jaccard_y0 = []
    jaccard_y1 = []
    jaccard_y2 = []
    jaccard_y3 = []

    for idx in indexes_y0:
        for neighbour in neighbours[idx.item()]:
            if neighbour != idx.item():
                jaccard_y0.append(_jaccard_similarity(data.x[idx], data.x[neighbour]))

    for idx in indexes_y1:
        for neighbour in neighbours[idx.item()]:
            if neighbour != idx.item():
                jaccard_y1.append(_jaccard_similarity(data.x[idx], data.x[neighbour]))

    for idx in indexes_y2:
        for neighbour in neighbours[idx.item()]:
            if neighbour != idx.item():
                jaccard_y2.append(_jaccard_similarity(data.x[idx], data.x[neighbour]))

    for idx in indexes_y3:
        for neighbour in neighbours[idx.item()]:
            if neighbour != idx.item():
                jaccard_y3.append(_jaccard_similarity(data.x[idx], data.x[neighbour]))



    return jaccard_y0, jaccard_y1, jaccard_y2, jaccard_y3

# Load the data (assuming `data` is already loaded)
data = load_data("inj_cora")
print(data.y)
# print the number of different values in data.y
print(np.unique(data.y.detach().numpy()))
#data.y = data.y.bool()

# Analyze the neighbours and get Jaccard similarities
jaccard_y0, jaccard_y1, jaccard_y2, jaccard_y3 = analyze_neighbours(data)

# Set the style
sns.set(style="whitegrid")

# Plot the box plot with more granular y-axis and prettier appearance
plt.figure(figsize=(8, 6))
sns.boxplot(data=[jaccard_y0, jaccard_y1, jaccard_y2, jaccard_y3], palette="Set3")
plt.ylabel('Jaccard Similarity', fontsize=14)
plt.title('Distribution of Jaccard Similarities (CORA)', fontsize=16)
plt.yticks(np.arange(0, 1.1, 0.05))  # Adjust the y-ticks to be more granular
plt.xlabel('Node Label', fontsize=14)
plt.xticks([0, 1, 2, 3], ['y=0 (benign)', 'y=1 (cont.)', 'y=2 (struct.)', 'y=3 (cont. & struct.)'], fontsize=12)
plt.grid(True)
plt.show()

# %%
