import scipy.sparse as sp
import numpy as np


class Nettack:
    """
    Nettack class used for poisoning attacks on node classification models.
    Copyright (C) 2018
    Daniel Zügner
    Technical University of Munich
    """

    def __init__(self,
                 adj: sp.csr_matrix,
                 X_obs: sp.csr_matrix,
                 z_obs: np.ndarray,
                 W1: np.ndarray,
                 W2: np.ndarray,
                 u: int,
                 verbose=False,
                 **kwargs):

        # Adjacency matrix
        self.adj = adj.copy().tolil()
        self.adj_no_selfloops = self.adj.copy()
        self.adj_no_selfloops.setdiag(0)
        self.adj_orig = self.adj.copy().tolil()
        self.u = u  # the node being attacked
        self.adj_preprocessed = preprocess_graph(self.adj).tolil()
        # Number of nodes
        self.N = adj.shape[0]

        # Node attributes
        self.X_obs = X_obs.copy().tolil()
        self.X_obs_orig = self.X_obs.copy().tolil()
        # Node labels
        self.z_obs = z_obs.copy()
        self.label_u = self.z_obs[self.u]
        self.K = np.max(self.z_obs) + 1
        # GCN weight matrices
        self.W1 = W1
        self.W2 = W2
        self.W = sp.csr_matrix(self.W1.T.dot(self.W2.T))

        self.cooc_matrix = self.X_obs.T.dot(self.X_obs).tolil()
        self.cooc_constraint = None

        self.structure_perturbations = []
        self.feature_perturbations = []

        self.influencer_nodes = []
        self.potential_edges = []
        self.verbose = verbose

        self.attr_adversary = None
        self.adj_adversary = None

    def compute_cooccurrence_constraint(self, nodes):
        """
        Co-occurrence constraint as described in the paper.

        Parameters
        ----------
        nodes: np.array
            Nodes whose features are considered for change

        Returns
        -------
        np.array [len(nodes), D], dtype bool
            Binary matrix of dimension len(nodes) x D. A 1 in entry n,d indicates that
            we are allowed to add feature d to the features of node n.

        """

        words_graph = self.cooc_matrix.copy()
        D = self.X_obs.shape[1]
        words_graph.setdiag(0)
        words_graph = (words_graph > 0)
        word_degrees = np.sum(words_graph, axis=0).A1

        inv_word_degrees = np.reciprocal(word_degrees.astype(float) + 1e-8)

        sd = np.zeros([self.N])
        for n in range(self.N):
            n_idx = self.X_obs[n, :].nonzero()[1]
            sd[n] = np.sum(inv_word_degrees[n_idx.tolist()])

        scores_matrix = sp.lil_matrix((self.N, D))

        for n in nodes:
            common_words = words_graph.multiply(self.X_obs[n])
            idegs = inv_word_degrees[common_words.nonzero()[1]]
            nnz = common_words.nonzero()[0]
            scores = np.array([idegs[nnz == ix].sum() for ix in range(D)])
            scores_matrix[n] = scores
        self.cooc_constraint = sp.csr_matrix(
            scores_matrix - 0.5 * sd[:, None] > 0)

    def gradient_wrt_x(self, label):
        """
        Compute the gradient of the logit belonging to the class of the input label with respect to the input features.

        Parameters
        ----------
        label: int
            Class whose logits are of interest

        Returns
        -------
        np.array [N, D] matrix containing the gradients.

        """

        return self.adj_preprocessed.dot(self.adj_preprocessed)[self.u].T.dot(self.W[:, label].T)

    def compute_logits(self):
        """
        Compute the logits of the surrogate model, i.e. linearized GCN.

        Returns
        -------
        np.array, [N, K]
            The log probabilities for each node.

        """
        return self.adj_preprocessed.dot(self.adj_preprocessed).dot(self.X_obs.dot(self.W))[self.u].toarray()[0]

    def strongest_wrong_class(self, logits):
        """
        Determine the incorrect class with largest logits.

        Parameters
        ----------
        logits: np.array, [N, K]
            The input logits

        Returns
        -------
        np.array, [N, L]
            The indices of the wrong labels with the highest attached log probabilities.
        """

        label_u_onehot = np.eye(self.K)[self.label_u]
        return (logits - 1000*label_u_onehot).argmax()
    


    def feature_scores(self):
        """
        Compute feature scores for all possible feature changes.
        """

        if self.cooc_constraint is None:
            self.compute_cooccurrence_constraint(self.influencer_nodes)
        logits = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits)
        gradient = self.gradient_wrt_x(
            self.label_u) - self.gradient_wrt_x(best_wrong_class)
        surrogate_loss = logits[self.label_u] - logits[best_wrong_class]

        gradients_flipped = (gradient * -1).tolil()
        gradients_flipped[self.X_obs.nonzero()] *= -1

        X_influencers = sp.lil_matrix(self.X_obs.shape)
        X_influencers[self.influencer_nodes] = self.X_obs[self.influencer_nodes]
        gradients_flipped = gradients_flipped.multiply(
            (self.cooc_constraint + X_influencers) > 0)
        nnz_ixs = np.array(gradients_flipped.nonzero()).T

        sorting = np.argsort(gradients_flipped[tuple(nnz_ixs.T)]).A1
        sorted_ixs = nnz_ixs[sorting]
        grads = gradients_flipped[tuple(nnz_ixs[sorting].T)]

        scores = surrogate_loss - grads
        return sorted_ixs[::-1], scores.A1[::-1]

    def struct_score(self, a_hat_uv, XW):
        """
        Compute structure scores, cf. Eq. 15 in the paper

        Parameters
        ----------
        a_hat_uv: sp.sparse_matrix, shape [P,2]
            Entries of matrix A_hat^2_u for each potential edge (see paper for explanation)

        XW: sp.sparse_matrix, shape [N, K], dtype float
            The class logits for each node.

        Returns
        -------
        np.array [P,]
            The struct score for every row in a_hat_uv
        """

        logits = a_hat_uv.dot(XW)
        label_onehot = np.eye(XW.shape[1])[self.label_u]
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:, self.label_u]
        struct_scores = logits_for_correct_class - best_wrong_class_logits

        return struct_scores

    def compute_XW(self):
        """
        Shortcut to compute the dot product of X and W
        Returns
        -------
        X.dot(W)
        """

        return self.X_obs.dot(self.W)

    def get_attacker_nodes(self, n=5, add_additional_nodes=False):
        """
        Determine the influencer nodes to attack node i based on the weights W and the attributes X.

        Parameters
        ----------
        n: int, default: 5
            The desired number of attacker nodes.

        add_additional_nodes: bool, default: False
            if True and the degree of node i (d_u) is < n, we select n-d_u additional attackers, which should
            get connected to u afterwards (outside this function).

        Returns
        -------
        np.array, shape [n,]:
            The indices of the attacker nodes.
        optional: np.array, shape [n - degree(n)]
            if additional_nodes is True, we separately
            return the additional attacker node indices

        """

        assert n < self.N - 1, "number of influencers cannot be >= number of nodes in the graph!"

        neighbors = self.adj_no_selfloops[self.u].nonzero()[1]
        assert self.u not in neighbors

        potential_edges = np.column_stack(
            (np.tile(self.u, len(neighbors)), neighbors)).astype("int32")

        # The new A_hat_square_uv values that we would get if we removed the edge from u to each of the neighbors,
        # respectively
        a_hat_uv = self.compute_new_a_hat_uv(potential_edges)

        XW = self.compute_XW()

        # compute the struct scores for all neighbors
        struct_scores = self.struct_score(a_hat_uv, XW).A1

        if len(neighbors) >= n:  # do we have enough neighbors for the number of desired influencers?
            influencer_nodes = neighbors[np.argsort(struct_scores)[:n]]
            if add_additional_nodes:
                return influencer_nodes, np.array([])
            return influencer_nodes
        else:
            influencer_nodes = neighbors
            if add_additional_nodes:  # Add additional influencers by connecting them to u first.
                # Compute the set of possible additional influencers, i.e. all nodes except the ones
                # that are already connected to u.
                poss_add_infl = np.setdiff1d(np.setdiff1d(
                    np.arange(self.N), neighbors), self.u)
                n_possible_additional = len(poss_add_infl)
                n_additional_attackers = n - len(neighbors)
                possible_edges = np.column_stack(
                    (np.tile(self.u, n_possible_additional), poss_add_infl))

                # Compute the struct_scores for all possible additional influencers, and choose the one
                # with the best struct score.
                a_hat_uv_additional = self.compute_new_a_hat_uv(possible_edges)
                additional_struct_scores = self.struct_score(
                    a_hat_uv_additional, XW)
                additional_influencers = poss_add_infl[np.argsort(
                    additional_struct_scores)[-n_additional_attackers::]]

                return influencer_nodes, additional_influencers
            else:
                return influencer_nodes

    def compute_new_a_hat_uv(self, potential_edges):
        """
        Compute the updated A_hat_square_uv entries that would result from inserting/deleting the input edges,
        for every edge.

        Parameters
        ----------
        potential_edges: np.array, shape [P,2], dtype int
            The edges to check.

        Returns
        -------
        sp.sparse_matrix: updated A_hat_square_u entries, a sparse PxN matrix, where P is len(possible_edges).
        """

        edges = np.array(self.adj.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = self.adj_preprocessed @ self.adj_preprocessed
        values_before = A_hat_sq[self.u].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1]
        twohop_ixs = np.array(A_hat_sq.nonzero()).T
        degrees = self.adj.sum(0).A1 + 1

        ixs, vals = compute_new_a_hat_uv(edges, node_ixs, edges_set, twohop_ixs, values_before, degrees,
                                         potential_edges, self.u)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])), shape=[
                                 len(potential_edges), self.N])

        return a_hat_uv

    def attack(self, n_perturbations, **kwargs):
        return self.attack_surrogate(n_perturbations, **kwargs)

    def attack_surrogate(self, n_perturbations, perturb_structure=True, perturb_features=True, avoid_edge_removal=False,
                         direct=True, n_influencers=0, delta_cutoff=0.004, **kwargs):
        """
        Perform an attack on the surrogate model.

        Parameters
        ----------
        n_perturbations: int
            The number of perturbations (structure or feature) to perform.

        perturb_structure: bool, default: True
            Indicates whether the structure can be changed.

        perturb_features: bool, default: True
            Indicates whether the features can be changed.

        direct: bool, default: True
            indicates whether to directly modify edges/features of the node attacked or only those of influencers.

        n_influencers: int, default: 0
            Number of influencing nodes -- will be ignored if direct is True

        delta_cutoff: float
            The critical value for the likelihood ratio test of the power law distributions.
             See the Chi square distribution with one degree of freedom. Default value 0.004
             corresponds to a p-value of roughly 0.95.

        Returns
        -------
        None.

        """

        assert not (not direct and n_influencers == 0), "indirect mode requires at least one influencer node"
        assert n_perturbations > 0, "need at least one perturbation"
        assert perturb_features or perturb_structure, "either perturb_features or perturb_structure must be true"

        logits_start = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits_start)
        surrogate_losses = [logits_start[self.label_u] -
                            logits_start[best_wrong_class]]

        if self.verbose:
            logging.info("##### Starting attack #####")
            if perturb_structure and perturb_features:
                logging.info(
                    "##### Attack node with ID {} using structure and feature perturbations #####".format(self.u))
            elif perturb_features:
                logging.info("##### Attack only using feature perturbations #####")
            elif perturb_structure:
                logging.info("##### Attack only using structure perturbations #####")
            if direct:
                logging.info("##### Attacking the node directly #####")
            else:
                logging.info("##### Attacking the node indirectly via {} influencer nodes #####".format(
                    n_influencers))
            logging.info("##### Performing {} perturbations #####".format(n_perturbations))

        if perturb_structure:

            # Setup starting values of the likelihood ratio test.
            degree_sequence_start = self.adj_orig.sum(0).A1
            current_degree_sequence = self.adj.sum(0).A1
            d_min = 2
            S_d_start = np.sum(
                np.log(degree_sequence_start[degree_sequence_start >= d_min]))
            current_S_d = np.sum(
                np.log(current_degree_sequence[current_degree_sequence >= d_min]))
            n_start = np.sum(degree_sequence_start >= d_min)
            current_n = np.sum(current_degree_sequence >= d_min)
            alpha_start = compute_alpha(n_start, S_d_start, d_min)
            log_likelihood_orig = compute_log_likelihood(
                n_start, alpha_start, S_d_start, d_min)

        if len(self.influencer_nodes) == 0:
            if not direct:
                # Choose influencer nodes
                infls, add_infls = self.get_attacker_nodes(
                    n_influencers, add_additional_nodes=True)
                self.influencer_nodes = np.concatenate(
                    (infls, add_infls)).astype("int")
                # Potential edges are all edges from any attacker to any other node, except the respective
                # attacker itself or the node being attacked.
                self.potential_edges = np.row_stack([np.column_stack((np.tile(infl, self.N - 2),
                                                                      np.setdiff1d(np.arange(self.N),
                                                                                   np.array([self.u, infl]))
                                                                      )) for infl in self.influencer_nodes])
                if self.verbose:
                    logging.info("Influencer nodes: {}".format(self.influencer_nodes))
            else:
                # direct attack
                influencers = [self.u]
                self.potential_edges = np.column_stack(
                    (np.tile(self.u, self.N - 1), np.setdiff1d(np.arange(self.N), self.u)))
                self.influencer_nodes = np.array(influencers)
        self.potential_edges = self.potential_edges.astype("int32")
        if avoid_edge_removal:
            self.potential_edges = self.potential_edges[
                self.adj_orig[self.potential_edges[:, 0], self.potential_edges[:, 1]].toarray().reshape(-1) == 0
            ]
        for _ in tqdm(range(n_perturbations)):
            if perturb_structure:

                # Do not consider edges that, if removed, result in singleton edges in the graph.
                singleton_filter = filter_singletons(
                    self.potential_edges, self.adj)
                filtered_edges = self.potential_edges[singleton_filter]

                # Update the values for the power law likelihood ratio test.
                deltas = 2 * \
                    (1 - self.adj[tuple(filtered_edges.T)].toarray()[0]) - 1
                d_edges_old = current_degree_sequence[filtered_edges]
                d_edges_new = current_degree_sequence[filtered_edges] + \
                    deltas[:, None]
                new_S_d, new_n = update_Sx(
                    current_S_d, current_n, d_edges_old, d_edges_new, d_min)
                new_alphas = compute_alpha(new_n, new_S_d, d_min)
                new_ll = compute_log_likelihood(
                    new_n, new_alphas, new_S_d, d_min)
                alphas_combined = compute_alpha(
                    new_n + n_start, new_S_d + S_d_start, d_min)
                new_ll_combined = compute_log_likelihood(
                    new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min)
                new_ratios = -2 * new_ll_combined + \
                    2 * (new_ll + log_likelihood_orig)

                # Do not consider edges that, if added/removed, would lead to a violation of the
                # likelihood ration Chi_square cutoff value.
                powerlaw_filter = filter_chisquare(new_ratios, delta_cutoff)
                filtered_edges_final = filtered_edges[powerlaw_filter]

                # Compute new entries in A_hat_square_uv
                a_hat_uv_new = self.compute_new_a_hat_uv(filtered_edges_final)
                # Compute the struct scores for each potential edge
                struct_scores = self.struct_score(
                    a_hat_uv_new, self.compute_XW())
                best_edge_ix = struct_scores.argmin()
                best_edge_score = struct_scores.min()
                best_edge = filtered_edges_final[best_edge_ix]

            if perturb_features:
                # Compute the feature scores for each potential feature perturbation
                feature_ixs, feature_scores = self.feature_scores()
                best_feature_ix = feature_ixs[0]
                best_feature_score = feature_scores[0]

            if perturb_structure and perturb_features:
                # decide whether to choose an edge or feature to change
                if best_edge_score < best_feature_score:
                    if self.verbose:
                        logging.info("Edge perturbation: {}".format(best_edge))
                    change_structure = True
                else:
                    if self.verbose:
                        logging.info("Feature perturbation: {}".format(best_feature_ix))
                    change_structure = False
            elif perturb_structure:
                change_structure = True
            elif perturb_features:
                change_structure = False

            if change_structure:
                # perform edge perturbation

                self.adj[tuple(best_edge)] = self.adj[tuple(
                    best_edge[::-1])] = 1 - self.adj[tuple(best_edge)]

                self.adj_preprocessed = preprocess_graph(self.adj)

                self.structure_perturbations.append(tuple(best_edge))
                self.feature_perturbations.append(())
                surrogate_losses.append(best_edge_score)

                # Update likelihood ratio test values
                current_S_d = new_S_d[powerlaw_filter][best_edge_ix]
                current_n = new_n[powerlaw_filter][best_edge_ix]
                current_degree_sequence[best_edge] += deltas[powerlaw_filter][best_edge_ix]

            else:
                self.X_obs[tuple(best_feature_ix)] = 1 - \
                    self.X_obs[tuple(best_feature_ix)]

                self.feature_perturbations.append(tuple(best_feature_ix))
                self.structure_perturbations.append(())
                surrogate_losses.append(best_feature_score)
        self.attr_adversary = sparse_tensor(self.X_obs.tocoo())
        self.adj_adversary = sparse_tensor(self.adj_preprocessed.tocoo())

    def reset(self):
        """
        Reset Nettack
        """
        self.adj = self.adj_orig.copy()
        self.X_obs = self.X_obs_orig.copy()
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []
        self.cooc_constraint = None


def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized.astype(np.float32)
