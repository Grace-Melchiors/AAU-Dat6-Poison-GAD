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
