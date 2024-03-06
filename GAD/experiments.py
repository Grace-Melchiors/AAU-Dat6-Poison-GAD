from pygod.detector import DOMINANT
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from Utils.graph_utils import prepare_graph, adj_matrix_sparse_coo_to_dense
from Poison.local_dice import LocalDICE

def run_local_dice():
    data = Planetoid("./data/Cora", "Cora", transform=T.NormalizeFeatures())[0]
    detector = DOMINANT(hid_dim=64, num_layers=4, epoch=100)
    detector.fit(data)

    pred, score, prob, conf = detector.predict(data,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)


    node_attr, adj, labels = prepare_graph(data)
    ld = LocalDICE(adj=adj, attr=node_attr, labels=labels, target_node_id=2)
    ld._poison(n_perturbations=20, node_idx = 2)

    data_inj = Planetoid("./data/Cora", "Cora", transform=T.NormalizeFeatures())[0]

    # Convert adversary adjecency matrix into compatible dense tensor
    adj_adversary = adj_matrix_sparse_coo_to_dense(ld.adj_adversary)

    data_inj.edge_index = adj_adversary

    detector_inj = DOMINANT(hid_dim=64, num_layers=4, epoch=100)
    detector_inj.fit(data_inj)

    pred_inj, score_inj, prob_inj, conf_inj = detector_inj.predict(data_inj,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)

    

    return pred, score, prob, conf, pred_inj, score_inj, prob_inj, conf_inj




