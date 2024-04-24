import torch
import numpy as np

#Hypothesis: Anomalies will have unnatural weights
def compareWeights(model, anomaly_nodes):
    """
        Takes the attributes of each node and compares it with the weights the fitted model assigns

        model: a fitted model
        anomaly_nodes: the nodes which could be different from the rest

        returns:
            - List of effect a node has after 1st layer
            - List of effect a node has after 2nd layer
            - List of effect an anomaly has after 1st layer
            - List of effect an anomaly has after 2nd layer
    """
    weights1 = model.shared_encoder.gc1.lin.weight
    weights2 = model.shared_encoder.gc2.lin.weight

    attries = model.attrs

    layer1_scores = []
    layer2_scores = []
    layer1_scores_anom = []
    layer2_scores_anom = []

    # Add for layer 1
    for i, nodeatr in enumerate(attries):
        result = torch.matmul(weights1, nodeatr)
        layer1_scores.append(np.sum(result.detach().numpy()))

        if(i in anomaly_nodes):
            layer1_scores_anom.append(np.sum(result.detach().numpy()))

    # Add for layer 2
        result = torch.matmul(weights2, result)
        layer2_scores.append(np.sum(result.detach().numpy()))

        if(i in anomaly_nodes):
            layer2_scores_anom.append(np.sum(result.detach().numpy()))
    
    return layer1_scores, layer2_scores, layer1_scores_anom, layer2_scores_anom