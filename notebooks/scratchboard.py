# %%
import numpy as np
from pygod.utils import load_data
import os
import yaml
import torch
import torch.nn.functional as F


import torch_sparse as sp
from gad_adversarial_robustness.gad.dominant.dominant_cuda import Dominant
from gad_adversarial_robustness.utils.graph_utils import load_anomaly_detection_dataset
from gad_adversarial_robustness.poison.greedy import update_edge_index
from torch_geometric.utils import contains_self_loops, add_self_loops, add_remaining_self_loops, is_undirected, to_undirected

def update_edge_index(edge_index, changes, device):
    updated_edge_index = edge_index.clone().to(device)
    
    for change in changes:
        source, target, weight = change
        
        if weight == 1.0:  # Add edge
            new_edge = torch.tensor([[source, target], [target, source]], dtype=torch.long, device=device)
            updated_edge_index = torch.cat([updated_edge_index, new_edge], dim=1)
        elif weight == 0:  # Remove edge
            mask = ~(((updated_edge_index[0] == source) & (updated_edge_index[1] == target)) | ((updated_edge_index[0] == target) & (updated_edge_index[1] == source)))
            updated_edge_index = updated_edge_index[:, mask]

    return updated_edge_index

changes = [
    [2, 3, 0],
    [2, 6, 1]
]
edge_index_lol = torch.tensor([[2, 3, 4, 5, 6], [3, 2, 5, 4, 7]])
#Make undirected: 
print(to_undirected(edge_index_lol))

edge_index_lol = update_edge_index(edge_index_lol, changes, 'cpu')
print(edge_index_lol)



script_dir = os.path.abspath('')
yaml_path = os.path.join(script_dir, '..',  'configs', 'dominant_config.yaml')
with open(yaml_path) as file:
        config = yaml.safe_load(file)

dataset_caching_path = os.path.join(script_dir, '..', '..', '..', 'data')


dataset = load_data('inj_cora')
adj, _, _, adj_label = load_anomaly_detection_dataset(dataset, config['model']['device'])
#edge_index = torch.LongTensor(np.array(sp.coo_matrix(adj).nonzero()))
adj_label = torch.FloatTensor(adj_label).to(config['model']['device'])
#attrs = torch.FloatTensor(attrs)
print(dataset.edge_index.shape)

print("DATASET: INJ CORA")
print(f"Has self-loops: {contains_self_loops(dataset.edge_index)}")
print(f"Is undirected: {is_undirected(dataset.edge_index)}")

print(f'Contains: {contains_self_loops(dataset.edge_index)}')
edge_index = add_self_loops(dataset.edge_index)[0]
print(edge_index.shape)



def normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    row_sum = torch.sum(edge_index, dim=1)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[row_sum == 0] = 0
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    return torch.matmul(F.normalize(edge_index, p=2, dim=0), torch.matmul(d_mat_inv_sqrt, F.normalize(edge_index, p=2, dim=1)))

edge_index = normalize_edge_index(edge_index)
print("EDGE INDEX:")
print(edge_index.shape)
print(edge_index)

from torch_geometric.utils import to_torch_sparse_tensor, to_dense_adj, dense_to_sparse
edge_index = dense_to_sparse(torch.tensor(adj))[0].to(config['model']['device'])

#edge_index = to_torch_sparse_tensor(dataset.edge_index.to(config['model']['device']))
label = torch.Tensor(dataset.y.bool()).to(config['model']['device'])
attrs = dataset.x.to(config['model']['device'])

model = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                    device=config['model']['device'], edge_index=edge_index, adj_label=adj_label, attrs=attrs, label=label)
model.to(config['model']['device'])
model.fit(config, verbose=False)
edge_index_test = to_dense_adj(edge_index.to(config['model']['device']))[0]
print(edge_index_test)



valid_indices = torch.where(label == 1)[0]
valid_indices = valid_indices.detach().cpu().numpy()
sorted_indices = sorted(valid_indices, key=lambda i: model.score[i])
for i in sorted_indices:
    print(f'{i}: Label: {label[i]} Score:{model.score[i]}')
model.score[882]
label[882]
sorted_indices[:15]

sample_edge_index = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 1]]).to('cuda')
print(sample_edge_index.shape)
print(sample_edge_index)

changes = [
    #[2056, 2654, 0],
    #[2518, 2654, 0],
    [1257, 2518, 1.0],
    #[2126, 2654, 0],
    #[1732, 2654, 0],
    #[1119, 2126, 1.0]
]

sample_edge_index = update_edge_index(sample_edge_index, changes, 'cuda')
print(sample_edge_index)

changes = [
    #[2056, 2654, 0],
    #[2518, 2654, 0],
    [1257, 2518, 0],
    [1, 2, 0],
    #[2126, 2654, 0],
    #[1732, 2654, 0],
    #[1119, 2126, 1.0]
]

sample_edge_index = update_edge_index(sample_edge_index, changes, 'cuda')
print(sample_edge_index)


# %%
