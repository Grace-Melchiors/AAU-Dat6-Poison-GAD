from pygod.detector import DOMINANT
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, TUDataset
from Utils.graph_utils import prepare_graph, adj_matrix_sparse_coo_to_dense
from Utils.poison_utils import poison_n_nodes
from Utils.experiment_results import Experiment
from Poison.local_dice import LocalDICE
from pygod.metric import eval_roc_auc
import torch
from pygod.utils import load_data
import copy
from typing import Tuple, List, Any
from Utils.graph_utils import insert_anomalies
from torchtyping import TensorType


from sklearn.metrics import (
    roc_auc_score,
)


def run_local_dice() -> Tuple[Experiment, Experiment, List[int], Any]:
    #data = TUDataset(root="dataset/Mutag", name="MUTAG")[3]
    #data = insert_anomalies(data, ctx_n=2, ctx_k=3)
    # Todo, skip insert_anomalies if we use load_data function from PyGod
    #data = Planetoid("./data/Cora", "Cora", transform=T.NormalizeFeatures())[0]
    data = load_data("inj_cora")
    y_binary: List[int] = data.y.bool()

    #y_binary = data.y.tolist()
    #print(y_binary)

    detector = DOMINANT(hid_dim=64, num_layers=4, epoch=100)
    detector.fit(data)

    pred, score, prob, conf = detector.predict(data,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)
    experiment_before_poison = Experiment(data=data, pred=pred, prob=prob, score=score, conf=conf)

    node_attr, adj, labels = prepare_graph(data)
   
    ld = LocalDICE(adj=adj, attr=node_attr, labels=labels, target_node_id=2)
    # Todo: Vary perburtations
    #ld, node_idxs = poison_n_nodes(ld, 10, 75)
    ld, node_idxs = poison_n_nodes(ld, 3, 1)
    print("These are the node_idxs: ${}".format(node_idxs))

    # Convert adversary adjecency matrix into compatible dense tensor
    adj_adversary = adj_matrix_sparse_coo_to_dense(ld.adj_adversary)

    data_after_poison = copy.deepcopy(data)
    data_after_poison.edge_index = adj_adversary

    detector_poisoned = DOMINANT(hid_dim=64, num_layers=4, epoch=100)
    detector_poisoned.fit(data_after_poison)

    pred_after, score_after, prob_after, conf_after = detector_poisoned.predict(data_after_poison,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)
    experiment_after_poison = Experiment(data=data_after_poison, pred=pred_after, score=score_after, prob=prob_after, conf=conf_after)
    
    print("auc before poison:")
    print(roc_auc_score(y_binary, score))
    print("auc after poison:")
    print(roc_auc_score(y_binary, score_after))
    #print(roc_auc_score(data_inj.y.detach().numpy(), score_inj.detach.numpy()))
    
    return experiment_before_poison, experiment_after_poison, node_idxs, y_binary

    #return pred, score, prob, conf, pred_after, score_after, prob_after, conf_after


if __name__ == '__main__':
    run_local_dice()


