from Poison.base_classes import LocalPoison
from Poison.local_dice import LocalDICE
import random

# poisons n nodes and returns the id's of the nodes poisoned
def poison_n_nodes(poison_obj: LocalDICE, n_perturbations, n_nodes_to_poison):
    num_nodes = poison_obj.adj.size(0) # get number of nodes

    node_idxs = []
    for _ in range(n_nodes_to_poison):
        random_number = random.randint(0, num_nodes)
        node_idxs.append(random_number)
        poison_obj._poison(n_perturbations, random_number)

    # return the poison_obj, as well as the indeces of nodes that
    # have been poisoned
    return poison_obj, node_idxs





