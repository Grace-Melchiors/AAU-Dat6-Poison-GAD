import torch
from torch_sparse import SparseTensor
from Poison.base_classes import LocalPoison
class LocalDICE(LocalPoison):
    def __init__(self, add_ratio: float = 1.0, **kwargs):
        self.add_ratio = add_ratio
        self.adj_adversary = None

        super().__init__(**kwargs)

    def _poison(self, n_perturbations: int, node_idx: int, **kwargs):


        # we first remove the edges that connect to the nodes of the same class of the targeted node
        # afterwards, we add edges that connect to nodes of a different class 
        # note that we sample the edges randomly


        add_budget = int(round((n_perturbations * self.add_ratio), 0)) #


        delete_budget = int(n_perturbations - add_budget)

        # step 1: we get the indices of all edges connected to target node
        adj_i = self.adj[node_idx]
        _, neighbors_idx, _ = adj_i.coo()

        # step 2: get the ones which connect to nodes of the same classes
        same_class_mask = self.labels[neighbors_idx] == self.labels[node_idx]

        # step 3: collect the nodes that are part of step 2, and exclude them from the potential new neighbor indeces
        exlude_from_add_idx = [node_idx] + neighbors_idx.tolist()
        add_neighbors_idx = self._sample_additions(
            node_idx,
            n_perturbations,
            min(delete_budget, same_class_mask.sum()),
            exclude=exlude_from_add_idx,
        )

        # 4. sample edges to nodes of step 2, and delete them
        delete_neighbors_mask = torch.full_like(neighbors_idx, False, dtype=bool)
        if delete_budget > 0:
            delete_neighbors_idx = neighbors_idx[same_class_mask][
                torch.randperm(same_class_mask.sum())
            ][:delete_budget]
            delete_neighbors_mask = (
                neighbors_idx.repeat(delete_neighbors_idx.shape[0]).view(
                    delete_neighbors_idx.shape[0], -1
                )
                == delete_neighbors_idx[:, None].repeat(1, neighbors_idx.shape[0])
            ).any(dim=0)

        # 5. build perturbed adjacency
        A_rows, A_cols, A_vals = self.adj.coo()
        A_idx = torch.stack([A_rows, A_cols], dim=0)

        is_before = A_rows < node_idx
        is_after = A_rows > node_idx

        i_col = (
            torch.cat([neighbors_idx[~delete_neighbors_mask], add_neighbors_idx], dim=0)
            .sort()
            .values
        )
        i_row = torch.full_like(i_col, node_idx)
        i_idx = torch.stack([i_row, i_col], dim=0)
        i_val = torch.ones(i_idx.shape[1])

        A_idx = torch.cat((A_idx[:, is_before], i_idx, A_idx[:, is_after]), dim=-1)
        A_weights = torch.cat((A_vals[is_before], i_val, A_vals[is_after]), dim=-1)

        self.perturbed_edges = i_idx
        self.adj_adversary = SparseTensor.from_edge_index(
            A_idx, A_weights, (self.n, self.n)
        )

    def _sample_additions(self, node_idx, n_perturbations, n_deletions, exclude=[]):
        """Randomly random sample edges from adjacency matrix, `exclude` is a set
        which contains the edges we do not want to sample and the ones already sampled
        """

        additions_idx = []
        while len(additions_idx) < n_perturbations - n_deletions:
            possible_edge = torch.randint(self.n, (1, 1)).item()
            if (
                possible_edge not in exclude
                and self.labels[possible_edge] != self.labels[node_idx]
            ):
                additions_idx.append(possible_edge)
                exclude.append(possible_edge)
        return torch.tensor(additions_idx, dtype=torch.long)


    def get_perturbed_edges(self) -> torch.Tensor:
        if not hasattr(self, "perturbed_edges"):
            return torch.tensor([])
        return self.perturbed_edges
