import pygod as pyg
import matplotlib.pyplot as plt

def get_all_eval_metrics(label, score, pred, k=None):
    "Returns avg_prec, f1, prec_k, recall_k, roc_auc"
    
    avg_prec = pyg.metric.eval_average_precision(label, score)
    f1 = pyg.metric.eval_f1(label, pred)
    prec_k = pyg.metric.eval_precision_at_k(label, score)
    recall_k = pyg.metric.eval_recall_at_k(label, score)
    roc_auc = pyg.metric.eval_roc_auc(label, score)
    
    return avg_prec, f1, prec_k, recall_k, roc_auc



def perturbStats(perturb, amomaly_list):
    """
        returns: 
            AtA_del_count: Anomaly to Anomaly deletion count
            AtA_add_count: Anomaly to Anomaly addition count
            AtN_del_count: Anomaly to Normal deletion count
            AtN_add_count: Anomaly to Normal addition count
            NtN_del_count: Normal to Normal deletion count
            NtN_add_count: Normal to Normal addition count

            del_dict: Dictionary of node indxs along with how often they have had deleted edges
            add_dict: Dictionary of node indxs along with how often they have had added edges
    """

    AtA_del_count = 0 #Anomaly to Anomaly deletion count
    AtA_add_count = 0 #Anomaly to Anomaly addition count
    AtN_del_count = 0 #Anomaly to Normal deletion count
    AtN_add_count = 0 #Anomaly to Normal addition count
    NtN_del_count = 0 #Normal to Normal deletion count
    NtN_add_count = 0 #Normal to Normal addition count

    del_dict = {}   # Dictionary of node indxs along with how often they have had deleted edges
    add_dict = {}   # Dictionary of node indxs along with how often they have had added edges
    
    def addToLib(index, dict_store):
        if index in dict_store:
            dict_store[index] += 1
        else:
            dict_store[index] = 1
    def addIndex(index, was_delete_action):
        if was_delete_action:
            addToLib(index, del_dict)
        else:
            addToLib(index, add_dict)

    for change in perturb:
        # Deletions
        if (change[0] in amomaly_list) and (change[1] in amomaly_list) and (change[2] == 1):
            AtA_del_count += 1
        elif ((change[0] in amomaly_list) and (change[1] not in amomaly_list) or 
            (change[0] not in amomaly_list) and (change[1] in amomaly_list)) and (change[2] == 1):
            AtN_del_count += 1
        elif (change[0] not in amomaly_list) and (change[1] not in amomaly_list) and (change[2] == 1):
            NtN_del_count += 1

        # Additions
        elif (change[0] in amomaly_list) and (change[1] in amomaly_list) and (change[2] == 0):
            AtA_del_count += 1
        elif ((change[0] in amomaly_list) and (change[1] not in amomaly_list) or 
            (change[0] not in amomaly_list) and (change[1] in amomaly_list)) and (change[2] == 0):
            AtN_del_count += 1
        elif (change[0] not in amomaly_list) and (change[1] not in amomaly_list) and (change[2] == 0):
            NtN_del_count += 1

        #Count up how often these nodes are deleted or added to
        addIndex(change[0], change[2])
        addIndex(change[1], change[2])

    return AtA_del_count, AtA_add_count, AtN_del_count, AtN_add_count, NtN_del_count, NtN_add_count, del_dict, add_dict

def draw_compare_clean_and_poison(clean_prob, poison_prob, label):
    """
        clean_prob: (long tensor) probabilities of clean nodes being anomalies
        poison_prob: (long tensor) probabilities of poison nodes being anomalies
        label: (long tensor) true labels of nodes being anomalies
        
        draws a graph comparing probabilities, highlighting true anomalies as red
        """
    
    # assign colors based on probability of being an outlier
    node_colors = []

    for i in range(label.numel()):
        if label[i].item() == 1: 
            node_colors.append('red')
        else:
            node_colors.append('blue')

    plt.scatter(clean_prob, poison_prob, marker='o', color=node_colors, label='Best')
    # axis labels   
    plt.xlabel('Clean Model', fontsize=26)
    plt.ylabel('Poisoned Model', fontsize=26)

    plt.show()
