# %%
import numpy as np
from pygod.utils import load_data
import os
import yaml
import torch


from gad_adversarial_robustness.gad.dominant.dominant_cuda import Dominant
from gad_adversarial_robustness.utils.graph_utils import load_anomaly_detection_dataset

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

from torch_geometric.utils import to_torch_sparse_tensor
edge_index = to_torch_sparse_tensor(dataset.edge_index.to(config['model']['device']))
label = torch.Tensor(dataset.y.bool()).to(config['model']['device'])
attrs = dataset.x.to(config['model']['device'])

model = Dominant(feat_size=attrs.size(1), hidden_size=config['model']['hidden_dim'], dropout=config['model']['dropout'],
                    device=config['model']['device'], edge_index=edge_index, adj_label=adj_label, attrs=attrs, label=label)
model.to(config['model']['device'])
model.fit(config, verbose=False)

valid_indices = torch.where(label == 1)[0]
valid_indices = valid_indices.detach().cpu().numpy()
sorted_indices = sorted(valid_indices, key=lambda i: model.score[i])
for i in sorted_indices:
    print(f'{i}: Label: {label[i]} Score:{model.score[i]}')
model.score[882]
label[882]
sorted_indices[:15]

# %%
