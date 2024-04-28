# %% 
from scipy.io import loadmat
import numpy as np
import scipy.sparse as sp
from pygod.utils import load_data
import torch
from torch_geometric.data import Data


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


data: Data = load_data("inj_cora")
#loaded_edge_idx = torch.load("../../notebooks/276_budget_greedy_edge_index.pt")
#data.edge_index = loaded_edge_idx
data.y = data.y.bool()

mode = 'pos'  # if set to pos, it only compute two metrics for positive nodes

label = data.y.cpu().numpy() if torch.is_tensor(data.y) else data.y

# Extract the edges of positive nodes in each relation graph
pos_nodes = set((label.nonzero()[0]).tolist())
edge_index = data.edge_index.cpu().numpy() if torch.is_tensor(data.edge_index) else data.edge_index

# Assuming 'edge_index' is of shape [2, num_edges] and type 'numpy.ndarray'
node_list = [set(edge_index[0].tolist())]
pos_node_list = [list(net_nodes.intersection(pos_nodes)) for net_nodes in node_list]
pos_idx_list = []
for pos_node in pos_node_list:
    pos_idx_list.append(np.in1d(edge_index[0], np.array(pos_node)).nonzero()[0])

# Assuming 'data.x' is a tensor of node features
feature = data.x.cpu().numpy() if torch.is_tensor(data.x) else data.x


feature = normalize(feature)

feature_simi_list = []
label_simi_list = []
print('compute two metrics')
for pos_idx in pos_idx_list:
    feature_simi = 0
    label_simi = 0
    if mode == 'pos':  # compute two metrics for positive nodes
        for idx in pos_idx:
            u, v = edge_index[:, idx]
            feature_simi += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
            label_simi += label[u] == label[v]

        feature_simi = feature_simi / len(pos_idx)
        label_simi = label_simi / len(pos_idx)

    else:  # compute two metrics for all nodes
        for u, v in edge_index.T:
            feature_simi += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
            label_simi += label[u] == label[v]

        feature_simi = feature_simi / edge_index.shape[1]
        label_simi = label_simi / edge_index.shape[1]

    feature_simi_list.append(feature_simi)
    label_simi_list.append(label_simi)

print(f'feature_simi: {feature_simi_list}')
print(f'label_simi: {label_simi_list}')

# %%
