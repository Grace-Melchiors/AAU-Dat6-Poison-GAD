import torch.nn as nn
import torch.nn.functional as F
import torch

from dominant_Layers import GraphConvolution

# the encoder part of the autoencoder
class Encoder(nn.Module):
    """
    encodes the input features using graph convolutional layers
        nfeat: node feature size
        nhid: hidden layer size
        dropout: dropout rate

    purpose: 
        aims to transform input features into a latent representation by levraging GC layers, 
        which are crucial for processing graph structured data
    """
    def __init__(self, nfeat, nhid, dropout) :
        # refers to the fact that this is a subclass of nn.Module and is inheriting all methods thereof
        super(Encoder, self).__init__()

        # on initialization, creates 2 GCN layers -> goes from nfeat size to nhid
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 =  GraphConvolution(nhid, nhid)

        self.dropout = dropout

    def forward(self, x, adj):
        """
        Defines the computation flow during the forward pass
            x: input features
        """
        # Applies ReLU activation fxn after each GCN layer (intrduces non-linearity)
        x = F.relu(self.gc1(x, adj))
        # applies dropout regularization (prevent overfitting)
        x = F.dropout(x, self.dropout, training=self.training) 
        # Passes the processed features through the second GC layer
        x = F.relu(self.gc2(x, adj))
        
        return x #returns the final encoding of input features
    
    # plays a crucial role in the NN architecture by encoding input features into a 
    # meaningful latent space using graph convolutional operations, enabling the model to effectively 
    # learn and extract valuable information from graph-structured data.


class Attribute_Decoder(nn.Module):
    """
    Responsible for decoding features back to their original space using graph convolutional layers
    i.e. it is designed to reverse the encoding process performed by the Enocoder - reconstructing the 
    original features from the encoded representation

        nfeat: node feature size
        nhid: hidden layer size
        dropout: dropout rate for regularization
    """
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        # initialization it creates 2 GC layers
        self.gc1 = GraphConvolution(nhid, nhid) # dimensions nhid x nhid 
        self.gc2 = GraphConvolution(nhid, nfeat) # dimensions nhid x nfeat, because it is increasing in size from gc1 to gc2

        self.dropout = dropout


    def forward(self, x, adj):
        """
        Defines the decoding process during the forward pass
            x: input features
        """
        # Applies ReLU activation fxn after each GCN layer (intrduces non-linearity)
        x = F.relu(self.gc1(x, adj))
        # applies dropout regularization (prevent overfitting)
        x = F.dropout(x, self.dropout, training=self.training) 
        # Passes the processed features through the second GC layer
        x = F.relu(self.gc2(x, adj))
        
        return x # returns the decoded features

class Structure_Decoder(nn.Module):
    """
    Responsible for decoding the structural information of the graph back to its original form
    -> i.e. focuses on reconstructing the original structural information of the graph from the learned latent representation 
    """
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        Decoding process for structural information
        """
        x = F.relu(self.gc1(x, adj))
        # dropout step to prevent overfitting 
        x = F.dropout(x, self.dropout, training=self.training) 
        # Computes the adjacency matrix by multiplying the output with its transpose
        #   
        x = x @ x.T

        return x # returns the re-constructed adjacency matrix (represents strctural information)
    

class Dominant(nn.Module):
    """
    This class serves as the main model that integrates the shared encoder, attribute decoder and structure decoder components  
    """
    def __init__(self, feat_size, hidden_size, dropout):
        super(Dominant, self).__init__()

        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)

    def forward(self, x, adj):
        """
        combines the components above
        """
        # encode
        x = self.shared_encoder(x, adj)

        # decode feature matrix
        x_hat = self.attr_decoder(x, adj)

        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(x, adj)

        return struct_reconstructed, x_hat
    



