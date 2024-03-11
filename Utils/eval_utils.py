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




def draw_compare_clean_and_poison(clean_prob, poison_prob, label):
    """
        clean_prob: (long tensor) probabilities of clean nodes being anomalies
        poison_prob: (long tensor) probabilities of poison nodes being anomalies
        label: (long tensor) true labels of nodes being anomalies
        
        draws a graph comparing probabilities, highlighting true anomalies as red
        """
    
    plt.scatter(clean_prob, poison_prob, marker='o', color='black', label='Best')
    # axis labels   
    plt.xlabel('Clean Model', fontsize=26)
    plt.ylabel('Poisoned Model', fontsize=26)

    plt.show()
