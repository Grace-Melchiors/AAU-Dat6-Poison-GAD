import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module): 
    """ 
    Simple GCN Layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True): 
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # assigns a float tensor of the dimensions in_feature by out_feature - will be used to store the weights of a neural network layer
        # wrapping the tensor in a "Parameter" object, allows it to be automatically included in the list of trainable model parameters
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            # if bias = true, create a tensor of length: "out_features" and wrap it in the Parameter object to be included in training
            self.bias = Parameter(torch.FloatTensor(out_features))
        else: 
            # if bias = false, register it as none
            self.register_parameter('bias', None)

        self.reset_parameters()

    
    def reset_parameters(self):
        """
        used to initialize the weights and biases of the nn layer 
        by using a random uniform distribution centered around zero, 
        with a standard deviation calculated based on the number of input features.
        """

        # calculates the standard deviation for initizalizing weights 
        #   - the standard deviation is given by math.sqrt(n), where n is the number of input features
        standardDeviation = 1. / math.sqrt(self.weight.size(1))

        # initializes the weights of the layer with random numbers drawn from a uniform distribution between -stdv and stdv
        #   This helps ensure that the weights start with small random values centered around zero - can help improve convergence during training
        self.weight.data.uniform_(-standardDeviation, standardDeviation)

        if self.bias is not None: 
            # similarly to the weights, initializes bias to random numbers drawn from the uniform distribution btween -stdv and stdv
            self.bias.data.uniform_(-standardDeviation, standardDeviation)

    def forward(self, input, adj):
        """
        defines the forward propagation logic for the layer

        input: This represents the input to the layer.
        adj: represents some form of adjacency matrix or graph structure.
        """        

        # Performs a linear transformation
        #   matrix multiplication between the input tensor and weight tensor, i.e. linear transform of the input features by the learnable weights
        support = torch.mm(input, self.weight)

        # performs sparse matrix multiplication and generates the final output of the graph convolutional layer
        #   i.e. Utilize sparse matrix multiplication with the adjacency matrix to incorporate neighborhood information
        output = torch.spmm(adj, support)

        # adds bias to the output if it exists
        if self.bias is not None:   
            return output + self.bias
        else:
            return output
        

    def __repr__(self):
        """
        responsible for defining a string representation of the class instance
            returns a string that represents the class and its attributes 
        """
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


