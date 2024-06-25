# %%
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

LOAD = True
if LOAD:
    loaded_values_1 = load_results('results_citeseer_unmodified_and_camoblock.pkl')
    loaded_values_2 = load_results('results_citeseer_jaccard_medoid.pkl')
    #loaded_values_1 = load_results('./notebooks/results_citeseer_unmodified_and_camoblock.pkl')
    #loaded_values_2 = load_results('./notebooks/results_citeseer_jaccard_medoid.pkl')
    AS_1, AS_DOM_1, AUC_DOM_1, ACC_DOM_1, perturb_1, edge_index_1, CHANGE_IN_TARGET_NODE_AS_1, LAST_FEAT_LOSS, LAST_STRUCT_LOSS, ALL_AS = loaded_values_1
    AS_2, AS_DOM_2, AUC_DOM_2, ACC_DOM_2, perturb_2, edge_index_2, CHANGE_IN_TARGET_NODE_AS_2, LAST_FEAT_LOSS_2, LAST_STRUCT_LOSS_2, ALL_AS_2 = loaded_values_2

# %%
from matplotlib import pyplot as plt
def plot_anomaly_scores(anomaly_scores, model_name, additional_scores=None, title=None):
    iterations = range(len(anomaly_scores))  # Number of iterations

    # Plot the primary anomaly scores
    for node_index in range(len(anomaly_scores[0])):
        node_scores = [scores[node_index] for scores in anomaly_scores]
        plt.plot(iterations, node_scores, label='DOMINANT w/ CamoBlock' if node_index == 0 else None)

    # Plot the additional anomaly scores if provided
    if additional_scores is not None:
        for node_index in range(len(additional_scores[0])):
            node_scores = [scores[node_index] for scores in additional_scores]
            plt.plot(iterations, node_scores, linestyle=':', label='Unmodified DOMINANT' if node_index == 0 else None)

    plt.xlabel('Budget')
    plt.ylabel('Anomaly Score')
    if title is None:
        plt.title(f'Anomaly Score Development for Each Node: {model_name}')
    else:
        plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.show()



plot_anomaly_scores(CHANGE_IN_TARGET_NODE_AS_1[1], "CamoBlock", CHANGE_IN_TARGET_NODE_AS_1[0], "Anomaly Score Development for Each Node (CiteSeer)")
# %%


def plot_scores(scores1, scores2, title='AUC Scores by Budget', xlabel='Budget', ylabel='AUC Score', scores3=None, scores4=None):
    """
    Plots a list of scores against their corresponding budgets.

    Parameters:
    - scores: List of scores to plot.
    - title: Title of the plot.
    - xlabel: Label for the X-axis.
    - ylabel: Label for the Y-axis.
    """
    budgets = range(0, len(scores1))

    # Creating the plot
    plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
    #plt.plot(budgets, scores, marker='o', linestyle='-', color='b')  # Plotting the scores
    plt.plot(budgets, scores1, marker='o', linestyle='-', color='b', label='Unmodified DOMINANT')  # Plotting the first set of scores
    plt.plot(budgets, scores2, marker='o', linestyle='-', color='r', label='DOMINANT w/ CamoBlock')  # Plotting the second set of scores
    if scores3 is not None:
        plt.plot(budgets, scores3, marker='o', linestyle='-', color='g', label='DOMINANT w/ JaccardGCN')  # Plotting the third set of scores
    if scores4 is not None:
        plt.plot(budgets, scores4, marker='o', linestyle='-', color='m', label='DOMINANT w/ Soft Medoid')  # Plotting the third set of scores

    # Adding some flair to the plot
    plt.title(title)  # Title of the plot
    plt.xlabel(xlabel)  # X-axis label
    plt.ylabel(ylabel)  # Y-axis label
    plt.grid(True)  # Adding a grid for better readability

    # Set integer ticks on the X-axis
    plt.xticks(range(0, 51,5))  # Set integer ticks

    plt.legend()

    # Display the plot
    plt.show()




plot_scores(AS_DOM_1[0], AS_DOM_1[1], "Sum of Target Nodes Anomaly Scores by Budget (CORA)", "Budget", "Anomaly Score", AS_DOM_2[0], AS_DOM_2[1])
plot_scores(AUC_DOM_1[0], AUC_DOM_1[1], "AUC-ROC by Budget (CORA)", "Budget", "AUC-ROC", AUC_DOM_2[0], AUC_DOM_2[1])
# %%
