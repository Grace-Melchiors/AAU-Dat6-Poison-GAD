# Read node features and labels from ".content" file
node_data = {}
with open('AnomalyDAE//cora//cora.content', 'r') as f:
    for line in f:
        parts = line.strip().split()
        node_id = parts[0]
        features = [float(x) for x in parts[1:-1]]  # Assuming features are numerical
        label = parts[-1]
        node_data[node_id] = {'features': features, 'label': label}

# Read edges from ".cites" file
edges = []
with open('AnomalyDAE//cora//cora.cites', 'r') as f:
    for line in f:
        source, target = line.strip().split()
        edges.append((source, target))

# use the 'node_data' dictionary and 'edges' list to construct the graph
# For example, using NetworkX:
import networkx as nx

graph = nx.Graph()
for node_id, data in node_data.items():
    graph.add_node(node_id, **data)

graph.add_edges_from(edges)

print(graph)
