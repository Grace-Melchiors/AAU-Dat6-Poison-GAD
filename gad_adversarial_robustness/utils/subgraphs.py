import torch
import random
import numpy as np
from datetime import datetime

from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes




def get_subset_neighbors(subset, edge_index, unique_neighbors=False):
    """ Subset (LongTensor): The nodes to get neighbors from.
        edge_index (LongTensor): The edge indices.
        unique_neighbors (bool): makes it such that there are no repeat neighbors
        
        returns (LongTensor): the list of nodes connected 
        
        Current implementation only tested with longtensors, potential bool problems
        """

    # Create mask to only select target edges
    if subset.dtype != torch.bool:
        num_nodes = maybe_num_nodes(edge_index) # gets amount of nodes present in edges
        node_mask = index_to_mask(subset, size=num_nodes) # creates node mask



    edge_mask = node_mask[edge_index[0]] # create edge mask to find nodes starting from origin

    edge_index = edge_index[:, edge_mask] # apply mask

    neighbor_nodes = edge_index[1]  # get the edge destinations 

    if unique_neighbors:
        neighbor_nodes = torch.unique(neighbor_nodes)

    return neighbor_nodes

def get_node_connectivity_RWR(node, edge_index, iterations = 100, restart_chance = 0.9, max_dist = None, random_seed = None):
    """ node ID (int): The node ID/number to get connectivity from.
        edge_index (LongTensor): The edge indices.
        iterations (int): amount of iterations to perform, one visit is an iteration.
        restart_chance (float): chance to restart to starting node
        max_dist (int): maximum amount of visist before forced restart
        random_seed (int): custom seed for random picking
        
        returns dictionary: dictionary of how frequently different nodes are visited
        """
    
    if random_seed is not int:
        random.seed(datetime.now().timestamp())
    else:
        random.seed(random_seed)

    connectivity = dict([])  
    originNode = node   #Keep track of the origin
    currentNode = node  #keep track of current

    # Main loop
    for i in range(iterations):
        # randomly pick next node
        neighbors = get_subset_neighbors(torch.tensor(currentNode), edge_index)
        neighborNumber = random.randint(0, neighbors.size()[0] - 1)
        nextNode = neighbors[neighborNumber].item()

        # Count the visit, unless it's origin
        if nextNode != originNode:
            if nextNode in connectivity:  #If seen before
                connectivity[nextNode] += 1
            else:                         # if new
                connectivity[nextNode] = 1
        
        # restart chance
        if random.uniform(0, 1) <= restart_chance:
            currentNode = originNode
        else:
            currentNode = nextNode
    
    return connectivity
        
    


def generate_RWR_subgraph(starting_node, edge_index, amount_of_nodes):
    """ starting_node (int): The node ID/number to get connectivity from.
        edge_index (LongTensor): The edge indices.
        amount_of_nodes (int): the amount of nodes to include in the subgraph, aside from original

        returns (list): a list of node ID's in the subgraph
        """
    connectivity = get_node_connectivity_RWR(starting_node, edge_index)

    # If fewer or equal nodes than [amount of nodes] to add, then add all
    if (len(connectivity) <= amount_of_nodes):
        return list(connectivity.keys())
    
    # Else sort and add top [amount of nodes]
    keys = list(connectivity.keys())
    values = list(connectivity.values())
    sorted_value_index = reversed(np.argsort(values))   #sort list ascending and reverse it to get descending

    connectivity = {keys[i]: values[i] for i in sorted_value_index} #order keys by sorted value indexes
    connectivity = list(connectivity)   # convert to list    
    connectivity = connectivity[0:amount_of_nodes]  # slice list
    connectivity.append(starting_node)  # add starting node

    return connectivity
