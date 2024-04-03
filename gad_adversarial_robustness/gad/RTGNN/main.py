import os
import argparse
import numpy as np
import torch
from gad_adversarial_robustness.gad.RTGNN.utils import noisify_with_P
from gad_adversarial_robustness.gad.RTGNN.dataset import Dataset
from gad_adversarial_robustness.gad.RTGNN.model.RTGNN import RTGNN

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12, help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--edge_hidden', type=int, default=64,
                    help='Number of hidden units of MLP graph constructor')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora",
                    choices=['cora', 'citeseer','blogcatalog'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.3,
                    help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=1,
                    help='loss weight of graph reconstruction')
parser.add_argument('--tau',type=float, default=0.05,
                    help='threshold of filtering noisy edges')
parser.add_argument('--th',type=float, default=0.95,
                    help='threshold of adding pseudo labels')
parser.add_argument("--K", type=int, default=100,
                    help='number of KNN search for each node')
parser.add_argument("--n_neg", type=int, default=100,
                    help='number of negitive sampling for each node')
parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'],
                    help='type of noises')
parser.add_argument('--decay_w', type=float, default=0.1,
                    help='down-weighted factor')
parser.add_argument('--co_lambda',type=float,default=0.1,
                     help='weight for consistency regularization term')



args = parser.parse_known_args()[0]
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = Dataset(root='./data', name=args.dataset)

adj, features, labels = data.adj, data.features, data.labels # initialize adj matrix, node features and labels

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test # initialize data split values: train, validation, test indexes

ptb = args.ptb_rate # noise perturbation rate --> it controls the amount of noise added to the training and validation labels

nclass = labels.max() + 1 # initalize number of classes
args.class_num=nclass

train_labels = labels[idx_train] # Extracting labels for training and validation data
val_labels = labels[idx_val]

train_val_labels = np.concatenate([train_labels,val_labels],axis=0) # concatenating values to np arrays
idx = np.concatenate([idx_train,idx_val],axis=0)


# Adding noise to the concatenated labels and getting the noise indices and clean indices
noise_y, P, noise_idx, clean_idx = noisify_with_P(train_val_labels,idx_train.shape[0],nclass, ptb, 10, args.noise)
args.noise_idx, args.clean_idx = noise_idx, clean_idx

noise_labels = labels.copy()
noise_labels[idx] = noise_y # set the noisy labels


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# instantiate RTGNN model
model = RTGNN(args, device)
model.fit(features, adj, noise_labels, labels, idx_train, idx_val, noise_idx, clean_idx) # train
print("===================")
model.test(idx_test) # test


