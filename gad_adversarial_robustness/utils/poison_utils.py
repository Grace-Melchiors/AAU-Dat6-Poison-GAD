from Poison.base_classes import LocalPoison
from Poison.local_dice import LocalDICE
import random
import logging

logging.basicConfig(level=logging.DEBUG)

# poisons n nodes and returns the id's of the nodes poisoned
def poison_n_nodes(poison_obj: LocalPoison, n_perturbations, n_nodes_to_poison):
    num_nodes = poison_obj.adj.size(0) - 1 # get number of nodes
    logging.debug(f"There are {num_nodes} nodes in the graph")

    idxs_of_nodes_poisoned = []
    for _ in range(n_nodes_to_poison):
        random_number = random.randint(0, num_nodes)
        print(f"node to atk {random_number}")
        idxs_of_nodes_poisoned.append(random_number)
        poison_obj.poison(n_perturbations, random_number)

    logging.debug(f"Nodes affected: {idxs_of_nodes_poisoned}")

    return poison_obj, idxs_of_nodes_poisoned





