#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2, 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.clustering.base import BaseClustering
from sknetwork.clustering.louvain_core import fit_core
from sknetwork.clustering.postprocess import reindex_labels
from sknetwork.utils.check import check_random_state, get_probs
from sknetwork.utils.format import get_adjacency, directed2undirected
from sknetwork.utils.membership import membership_matrix
from sknetwork.utils.verbose import VerboseMixin


class Louvain(BaseClustering, VerboseMixin):
    """Louvain algorithm for clustering graphs by maximization of modularity.

    Parameters
    ----------
    resolution :
        Resolution parameter.
    modularity : str
        Which objective function to maximize. Can be ``'dugue'``, ``'newman'`` or ``'potts'``.
    tol_optimization :
        Minimum increase in the objective function to enter a new optimization pass.
    tol_aggregation :
        Minimum increase in the objective function to enter a new aggregation pass.
    n_aggregations :
        Maximum number of aggregations.
        A negative value is interpreted as no limit.
    shuffle_nodes :
        Enables node shuffling before optimization.
    random_move :
        In the optimization phase, a random neighbor candidate is picked rather than looking at the whole neighborhood
    fast_move :
        In the optimization phase, only nodes whose neighborhood has been changed are visited
    refinement :
        Before the agregation phase, make an optimisation pass on each cluster separately
    sort_clusters :
        If ``True``, sort labels in decreasing order of cluster size.
    return_membership :
            If ``True``, return the membership matrix of nodes to each cluster (soft clustering).
    return_aggregate :
            If ``True``, return the adjacency matrix of the graph between clusters.
    random_state :
        Random number generator or random seed. If None, numpy.random is used.
    verbose :
        Verbose mode.

    Attributes
    ----------
    labels_ : np.ndarray
        Labels of the nodes.
    labels_row_ : np.ndarray
        Labels of the rows (for bipartite graphs).
    labels_col_ : np.ndarray
        Labels of the columns (for bipartite graphs).
    membership_ : sparse.csr_matrix
        Membership matrix of the nodes, shape (n_nodes, n_clusters).
    membership_row_ : sparse.csr_matrix
        Membership matrix of the rows (for bipartite graphs).
    membership_col_ : sparse.csr_matrix
        Membership matrix of the columns (for bipartite graphs).
    aggregate_ : sparse.csr_matrix
        Aggregate adjacency matrix or biadjacency matrix between clusters.

    Example
    -------
    >>> from sknetwork.clustering import Louvain
    >>> from sknetwork.data import karate_club
    >>> louvain = Louvain()
    >>> adjacency = karate_club()
    >>> labels = louvain.fit_transform(adjacency)
    >>> len(set(labels))
    4

    References
    ----------
    * Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
      `Fast unfolding of communities in large networks.
      <https://arxiv.org/abs/0803.0476>`_
      Journal of statistical mechanics: theory and experiment, 2008

    * Dugué, N., & Perez, A. (2015).
      `Directed Louvain: maximizing modularity in directed networks
      <https://hal.archives-ouvertes.fr/hal-01231784/document>`_
      (Doctoral dissertation, Université d'Orléans).

    * Traag, V. A., Van Dooren, P., & Nesterov, Y. (2011).
      `Narrow scope for resolution-limit-free community detection.
      <https://arxiv.org/pdf/1104.3083.pdf>`_
      Physical Review E, 84(1), 016114.

    * Traag, V. A. (2015).
      `Faster unfolding of communities: Speeding up the Louvain algorithm.
      <https://arxiv.org/pdf/1503.01322.pdf>`_
      Physical Review E, 92(3), 032801.

    * Ozaki, N., Tezuka, H., & Inaba, M. (2016).
      `A simple acceleration method for the Louvain algorithm.
      <http://www.ijcee.org/vol8/927-A023.pdf>`_
      International Journal of Computer and Electrical Engineering, 8(3), 207.

    * Waltman, L., & Van Eck, N. J. (2013).
      `A smart local moving algorithm for large-scale modularity-based community detection.
      <https://link.springer.com/content/pdf/10.1140/epjb/e2013-40829-0.pdf>`_
      The European physical journal B, 86(11), 1-14.
    """
    def __init__(self, resolution: float = 1, modularity: str = 'dugue', tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, n_aggregations: int = -1, shuffle_nodes: bool = False,
                 random_move: bool = False, fast_move: bool = False, refinement: bool = False,
                 sort_clusters: bool = True, return_membership: bool = True, return_aggregate: bool = True,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, verbose: bool = False):
        super(Louvain, self).__init__(sort_clusters=sort_clusters, return_membership=return_membership,
                                      return_aggregate=return_aggregate)
        VerboseMixin.__init__(self, verbose)

        self.resolution = resolution
        self.modularity = modularity.lower()
        self.tol = tol_optimization
        self.tol_aggregation = tol_aggregation
        self.n_aggregations = n_aggregations
        self.shuffle_nodes = shuffle_nodes
        self.random_move = random_move
        self.fast_move = fast_move
        self.refinement = refinement
        self.random_state = check_random_state(random_state)
        self.bipartite = None

    def _optimize(self, adjacency_norm, probs_ou, probs_in, labels):
        """One local optimization pass of the Louvain algorithm

        Parameters
        ----------
        adjacency_norm :
            the norm of the adjacency
        probs_ou :
            the array of degrees of the adjacency
        probs_in :
            the array of degrees of the transpose of the adjacency
        labels :
            the pre-existing labels of the nodes

        Returns
        -------
        labels :
            the communities of each node after optimization
        pass_increase :
            the increase in modularity gained after optimization
        """
        node_probs_in = probs_in.astype(np.float32)
        node_probs_ou = probs_ou.astype(np.float32)

        adjacency = 0.5 * directed2undirected(adjacency_norm)

        self_loops = adjacency.diagonal().astype(np.float32)

        indptr: np.ndarray = adjacency.indptr
        indices: np.ndarray = adjacency.indices
        data: np.ndarray = adjacency.data.astype(np.float32)
        labels: np.ndarray = labels.astype(type(indptr[0]))

        # fixing the random seed from the random state for the random neighbor case
        seed = self.random_state.get_state()[1][0] % 2**15

        return fit_core(self.resolution, self.tol, node_probs_ou, node_probs_in, self_loops,
                        data, indices, indptr, labels, self.random_move, self.fast_move, seed)

    @staticmethod
    def _aggregate(adjacency_norm, probs_ou, probs_in, membership: Union[sparse.csr_matrix, np.ndarray]):
        """Aggregate nodes belonging to the same cluster.

        Parameters
        ----------
        adjacency_norm :
            the normalized adjacency
        probs_ou :
            the array of degrees of the adjacency
        probs_in :
            the array of degrees of the transpose of the adjacency
        membership :
            membership matrix (rows).

        Returns
        -------
        Aggregate graph.
        """
        adjacency_norm = (membership.T.dot(adjacency_norm.dot(membership))).tocsr()
        probs_in = np.array(membership.T.dot(probs_in).T)
        probs_ou = np.array(membership.T.dot(probs_ou).T)
        return adjacency_norm, probs_ou, probs_in

    def _refine(self, adjacency_norm, labels_cluster):
        """Aggregate nodes belonging to the same cluster.

        Parameters
        ----------
        adjacency_norm :
            the normalized adjacency
        labels_cluster :
            the labels obtained after the optimization phase

        Returns
        -------
        Refined labels and corresponding reduced labels from the input
        """
        n_clusters = max(labels_cluster) + 1
        if n_clusters == 1:
            return labels_cluster, np.zeros(1, dtype=int)
        elif n_clusters == adjacency_norm.shape[0]:
            return labels_cluster, np.arange(n_clusters, dtype=int)
        else:
            current_max = -1
            labels_refined = np.zeros(adjacency_norm.shape[0], dtype=int)
            merged_labels_cluster = []
            for cluster in range(n_clusters):
                mask = labels_cluster == cluster
                sub_adjacency = adjacency_norm[mask, :][:, mask]
                local_labels_refined = np.arange(sub_adjacency.shape[0], dtype=int)
                if sub_adjacency.data.any():
                    if self.modularity == 'potts':
                        sub_probs_ou = get_probs('uniform', sub_adjacency)
                        sub_probs_in = sub_probs_ou.copy()
                    elif self.modularity == 'newman':
                        sub_probs_ou = get_probs('degree', sub_adjacency)
                        sub_probs_in = sub_probs_ou.copy()
                    elif self.modularity == 'dugue':
                        sub_probs_ou = get_probs('degree', sub_adjacency)
                        sub_probs_in = get_probs('degree', sub_adjacency.T)
                    else:
                        raise ValueError('Unknown modularity function.')
                    sub_adjacency /= sub_adjacency.data.sum()
                    local_labels_refined, _ = self._optimize(sub_adjacency, sub_probs_ou, sub_probs_in,
                                                             local_labels_refined)
                unique_clusters, local_labels_refined = np.unique(local_labels_refined, return_inverse=True)
                local_labels_refined += current_max + 1
                merged_labels_cluster += len(unique_clusters) * [cluster]
                labels_refined[mask] = local_labels_refined
                current_max = max(local_labels_refined)
            return labels_refined, np.array(merged_labels_cluster)


    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], force_bipartite: bool = False) -> 'Louvain':
        """Fit algorithm to the data.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        force_bipartite :
            If ``True``, force the input matrix to be considered as a biadjacency matrix even if square.

        Returns
        -------
        self: :class:`Louvain`
        """
        self._init_vars()

        if self.modularity == 'dugue':
            adjacency, self.bipartite = get_adjacency(input_matrix, force_directed=True,
                                                      force_bipartite=force_bipartite)
        else:
            adjacency, self.bipartite = get_adjacency(input_matrix, force_bipartite=force_bipartite)

        n = adjacency.shape[0]

        if self.modularity == 'potts':
            probs_ou = get_probs('uniform', adjacency)
            probs_in = probs_ou.copy()
        elif self.modularity == 'newman':
            probs_ou = get_probs('degree', adjacency)
            probs_in = probs_ou.copy()
        elif self.modularity == 'dugue':
            probs_ou = get_probs('degree', adjacency)
            probs_in = get_probs('degree', adjacency.T)
        else:
            raise ValueError('Unknown modularity function.')

        nodes = np.arange(n)
        if self.shuffle_nodes:
            nodes = self.random_state.permutation(nodes)
            adjacency = adjacency[nodes, :].tocsc()[:, nodes].tocsr()

        adjacency_cluster = adjacency / adjacency.data.sum()

        membership = sparse.identity(n, format='csr')
        increase = True
        count_aggregations = 0
        labels_cluster = np.arange(adjacency_cluster.shape[0], dtype=int)
        self.log.print("Starting with", n, "nodes.")
        while increase:
            count_aggregations += 1

            labels_cluster, pass_increase = self._optimize(adjacency_cluster, probs_ou, probs_in, labels_cluster)
            _, labels_cluster = np.unique(labels_cluster, return_inverse=True)
            if pass_increase <= self.tol_aggregation:
                increase = False
            else:
                if self.refinement:
                    labels_refined, labels_cluster = self._refine(adjacency_cluster, labels_cluster)
                    if max(labels_refined) == 0:
                        break
                    membership_cluster = membership_matrix(labels_refined)
                    membership = membership.dot(membership_cluster)
                    adjacency_cluster, probs_ou, probs_in = self._aggregate(adjacency_cluster, probs_ou, probs_in,
                                                                            membership_cluster)
                else:
                    membership_cluster = membership_matrix(labels_cluster)
                    membership = membership.dot(membership_cluster)
                    adjacency_cluster, probs_ou, probs_in = self._aggregate(adjacency_cluster, probs_ou, probs_in,
                                                                            membership_cluster)
                    labels_cluster = np.arange(adjacency_cluster.shape[0], dtype=int)

                n = adjacency_cluster.shape[0]
                if n == 1:
                    break
            self.log.print("Aggregation", count_aggregations, "completed with", n, "clusters and ",
                           pass_increase, "increment.")
            if count_aggregations == self.n_aggregations:
                break

        if self.sort_clusters:
            labels = reindex_labels(membership.indices)
        else:
            labels = membership.indices
        if self.shuffle_nodes:
            reverse = np.empty(nodes.size, nodes.dtype)
            reverse[nodes] = np.arange(nodes.size)
            labels = labels[reverse]

        self.labels_ = labels
        if self.bipartite:
            self._split_vars(input_matrix.shape)
        self._secondary_outputs(input_matrix)

        return self
