from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        #print input_dim, output_dim and hidden_dim
        print(input_dim, hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def get_weights(self):
        return self.conv1.lin.weight, self.conv2.lin.weight

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPDecoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, decoder_hidden_dims, output_dim, nfeat, nclass, nhid):
        super(AutoEncoder, self).__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = MLPDecoder(latent_dim, decoder_hidden_dims, output_dim)
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_rec = self.decoder(z)
        return x_rec

    
    def get_encoder_weights(self):
        return self.encoder.get_weights()


class Trainer:
    def __init__(self, model, optimizer, num_epochs):
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs

class Trainer:
    def __init__(self, model, optimizer, num_epochs):
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train(self, x, edge_index, labels):
        for epoch in range(self.num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            x_rec = self.model(x, edge_index)
            loss = F.mse_loss(x_rec, x, reduction='none')
            loss_mean = torch.mean(loss)
            loss_mean.backward()
            self.optimizer.step()

            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                self.model.eval()
                scores = torch.mean(loss, dim=1).detach().cpu().numpy()
                auc = roc_auc_score(labels.cpu().numpy(), scores)
                print(f"Epoch: {epoch:04d}, AUC-ROC: {auc:.4f}")

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss_mean.item():.4f}")