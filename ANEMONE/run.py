from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse

from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/Dataset/Cora', name='Cora')

data = dataset[0]
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')




os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

#Following what is recommended for CORA from https://github.com/GRAND-Lab/ANEMONE/blob/main/run.py
EXPID = 0.8
DATASET = 'cora'
LR = 1e-3
WEIGHT_DECAY = 0.0
AMOUNT_RUNS = 1
EMBEDDING_DIM = 64
PATIENCE = 100
NUM_EPOCH = 100
BATCH_SIZE = 300
SUBGRAPH_SIZE = 4
READOUT = 'avg'
AUC_TEST_ROUNDS = 256
NEGSAMP_RATIO_PATCH = 1
NEGSAMP_RATION_CONTEXT = 1
ALPHA = 1



if __name__ == '__main__':

    print('Dataset: {}'.format(DATASET), flush=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seeds = [i + 1 for i in range(AMOUNT_RUNS)]

    batch_size = BATCH_SIZE
    subgraph_size = SUBGRAPH_SIZE

    adj, features, labels, idx_train, idx_val,\
    idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(DATASET)


    


    features, _ = preprocess_features(features)
    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    all_auc = []
    for run in range(AMOUNT_RUNS):
        seed = seeds[run]
        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        dgl.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        model = Model(ft_size, EMBEDDING_DIM, 'prelu', NEGSAMP_RATIO_PATCH, NEGSAMP_RATION_CONTEXT,
                      READOUT).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
        b_xent_patch = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor([NEGSAMP_RATIO_PATCH]).to(device))
        b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor([NEGSAMP_RATION_CONTEXT]).to(device))

        cnt_wait = 0
        best = 1e9
        best_t = 0
        batch_num = nb_nodes // batch_size + 1

        for epoch in range(NUM_EPOCH):

            model.train()

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                lbl_patch = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * NEGSAMP_RATIO_PATCH))), 1).to(device)

                lbl_context = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * NEGSAMP_RATION_CONTEXT))), 1).to(device)

                ba = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                logits_1, logits_2 = model(bf, ba)

                # Context-level
                loss_all_1 = b_xent_context(logits_1, lbl_context)
                loss_1 = torch.mean(loss_all_1)

                # Patch-level
                loss_all_2 = b_xent_patch(logits_2, lbl_patch)
                loss_2 = torch.mean(loss_all_2)

                loss = ALPHA * loss_1 + (1 - ALPHA) * loss_2

                loss.backward()
                optimiser.step()

                loss = loss.detach().cpu().numpy()
                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'checkpoints/exp_{}.pkl'.format(EXPID))
            else:
                cnt_wait += 1

            if cnt_wait == PATIENCE:
                print('Early stopping!', flush=True)
                break

            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)

        # Testing
        print('Loading {}th epoch'.format(best_t), flush=True)
        model.load_state_dict(torch.load('checkpoints/exp_{}.pkl'.format(EXPID)))
        multi_round_ano_score = np.zeros((AUC_TEST_ROUNDS, nb_nodes))
        print('Testing AUC!', flush=True)

        for round in range(AUC_TEST_ROUNDS):
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
            for batch_idx in range(batch_num):
                optimiser.zero_grad()
                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]
                cur_batch_size = len(idx)
                ba = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                with torch.no_grad():
                    test_logits_1, test_logits_2 = model(bf, ba)
                    test_logits_1 = torch.sigmoid(torch.squeeze(test_logits_1))
                    test_logits_2 = torch.sigmoid(torch.squeeze(test_logits_2))

                if ALPHA != 1.0 and ALPHA != 0.0:
                    if NEGSAMP_RATION_CONTEXT == 1 and NEGSAMP_RATIO_PATCH == 1:
                        ano_score_1 = - (test_logits_1[:cur_batch_size] - test_logits_1[cur_batch_size:]).cpu().numpy()
                        ano_score_2 = - (test_logits_2[:cur_batch_size] - test_logits_2[cur_batch_size:]).cpu().numpy()
                    else:
                        ano_score_1 = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                            cur_batch_size, NEGSAMP_RATION_CONTEXT), dim=1)).cpu().numpy()  # context
                        ano_score_2 = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                            cur_batch_size, NEGSAMP_RATIO_PATCH), dim=1)).cpu().numpy()  # patch
                    ano_score = ALPHA * ano_score_1 + (1 - ALPHA) * ano_score_2
                elif ALPHA == 1.0:
                    if NEGSAMP_RATION_CONTEXT == 1:
                        ano_score = - (test_logits_1[:cur_batch_size] - test_logits_1[cur_batch_size:]).cpu().numpy()
                    else:
                        ano_score = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                                cur_batch_size, NEGSAMP_RATION_CONTEXT), dim=1)).cpu().numpy()  # context
                elif ALPHA == 0.0:
                    if NEGSAMP_RATIO_PATCH == 1:
                        ano_score = - (test_logits_2[:cur_batch_size] - test_logits_2[cur_batch_size:]).cpu().numpy()
                    else:
                        ano_score = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                                cur_batch_size, NEGSAMP_RATIO_PATCH), dim=1)).cpu().numpy()  # patch

                multi_round_ano_score[round, idx] = ano_score

        ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
        auc = roc_auc_score(ano_label, ano_score_final)
        all_auc.append(auc)
        print('Testing AUC:{:.4f}'.format(auc), flush=True)

    print('\n==============================')
    print(all_auc)
    print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)))
    print('==============================')