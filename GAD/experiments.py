from pygod.detector import DOMINANT
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from Utils.graph_utils import prepare_graph, adj_matrix_sparse_coo_to_dense
from Utils.poison_utils import poison_n_nodes
from Poison.local_dice import LocalDICE
from pygod.metric import eval_roc_auc
import torch
from pygod.utils import load_data

import sys

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)


def run_local_dice():
    sys.path.append('..')
    #data = Planetoid("./data/Cora", "Cora", transform=T.NormalizeFeatures())[0]
    data = load_data("inj_cora")
    y_binary = data.y.bool()

    detector = DOMINANT(hid_dim=64, num_layers=4, epoch=100)
    detector.fit(data)

    pred, score, prob, conf = detector.predict(data,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)

    node_attr, adj, labels = prepare_graph(data)
   
    ld = LocalDICE(adj=adj, attr=node_attr, labels=labels, target_node_id=2)
    # Todo: Vary perburtations
    ld, node_idxs = poison_n_nodes(ld, 5, 100)
    print("These are the node_idxs: ${}".format(node_idxs))




    # Convert adversary adjecency matrix into compatible dense tensor
    adj_adversary = adj_matrix_sparse_coo_to_dense(ld.adj_adversary)

    data.edge_index = adj_adversary

    detector_poisoned = DOMINANT(hid_dim=64, num_layers=4, epoch=100)
    detector_poisoned.fit(data)

    pred_after, score_after, prob_after, conf_after = detector_poisoned.predict(data,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)
    
    print("auc before poison:")
    print(roc_auc_score(y_binary, score))
    print("auc after poison:")
    print(roc_auc_score(y_binary, score_after))
    #print(roc_auc_score(data_inj.y.detach().numpy(), score_inj.detach.numpy()))

    

    return pred, score, prob, conf, pred_after, score_after, prob_after, conf_after




