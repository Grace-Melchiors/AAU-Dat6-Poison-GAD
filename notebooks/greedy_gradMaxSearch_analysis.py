import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


from gad_adversarial_robustness.poison.greedy import multiple_AS, poison_attack



parser = argparse.ArgumentParser(description='')
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, help='Random seed.')
parser.add_argument('--dataset', type=str, default='blogcatalog', choices=['bitcoin_alpha','ca_grqc','wikivote','cora_ml','citeseer','blogcatalog'], help='dataset')
parser.add_argument('--trial', type=int, default=1, choices=[1,2,3,4,5], help='trial')
parser.add_argument('--size', type=int, default=10, choices=[10,30], help='target nodes size')
parser.add_argument('--B', type=int, default=35, choices=[35], help='budget B')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], help='device')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
root_dir = os.getcwd().replace('\\', '/')

A = np.loadtxt(root_dir + '/Datasets/'+args.dataset+'/adj.txt').astype('float')
n = len(A)

triple = []
for i in range(n):
    for j in range(i+1,n):
        triple.append([i,j,A[i,j]])
triple = np.array(triple)

target_node_lst = np.loadtxt(root_dir + '/Datasets/'+args.dataset+'/' + 'rand'+ str(args.size) + '_' + str(args.trial) + '.txt').astype('int')

mod_dir = root_dir + '/Datasets/'+args.dataset+'/GradMaxSearch/' + 'rand'+ str(args.size) + '_' + str(args.trial)



model = multiple_AS(target_lst = target_node_lst, n_node = n, device = 'cpu')
A_mod, AS = poison_attack(model, triple, args.B)
np.savetxt(mod_dir+'/AS.txt',AS)