#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 23, 2020
@author: Quentin Lutz <qlutz@enst.fr>
"""
from typing import Union, Optional

import numpy as np
from scipy import sparse
from scipy.special import softmax

from sknetwork.clustering.louvain import Louvain
from sknetwork.clustering.leiden_core import fit_core
from sknetwork.clustering.postprocess import reindex_labels
from sknetwork.utils.check import check_format, check_probs, check_square
from sknetwork.utils.format import directed2undirected
from sknetwork.utils.membership import membership_matrix


class Leiden(Louvain):
    """Leiden algorithm for clustering graphs by maximization of modularity.

    * Graphs
    * Digraphs

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
        Label of each node.
    membership_ : sparse.csr_matrix
        Membership matrix.
    adjacency_ : sparse.csr_matrix
        Adjacency matrix between clusters.

    Example
    -------
    >>> from sknetwork.clustering import Louvain
    >>> from sknetwork.data import karate_club
    >>> leiden = Leiden()
    >>> adjacency = karate_club()
    >>> labels = leiden.fit_transform(adjacency)
    >>> len(set(labels))
    4

    References
    ----------
    * Traag, V. A., Waltman, L., & van Eck, N. J. (2019).
      `From Louvain to Leiden: guaranteeing well-connected communities.
      <https://www.nature.com/articles/s41598-019-41695-z>`_
      Scientific reports, 9(1), 1-12.
    """

    def __init__(self, resolution: float = 1, random_factor = 1, modularity: str = 'dugue',
                 tol_optimization: float = 1e-3, tol_aggregation: float = 1e-3, n_aggregations: int = -1,
                 shuffle_nodes: bool = False, sort_clusters: bool = True, return_membership: bool = True,
                 return_aggregate: bool = True, random_state: Optional[Union[np.random.RandomState, int]] = None,
                 verbose: bool = False):
        super(Leiden, self).__init__(resolution=resolution, modularity=modularity, tol_optimization=tol_optimization,
                                     tol_aggregation=tol_aggregation, n_aggregations=n_aggregations,
                                     shuffle_nodes=shuffle_nodes, sort_clusters=sort_clusters,
                                     return_membership=return_membership, return_aggregate=return_aggregate,
                                     random_state=random_state, verbose=verbose)
        self.random_factor = random_factor

    def _optimize(self, adjacency_norm, probs_ou, probs_in, labels):
        """One local optimization pass of the Leiden algorithm

        Parameters
        ----------
        adjacency_norm :
            the norm of the adjacency
        probs_ou :
            the array of degrees of the adjacency
        probs_in :
            the array of degrees of the transpose of the adjacency

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

        indptr: np.ndarray = adjacency.indptr.astype(np.int32)
        indices: np.ndarray = adjacency.indices.astype(np.int32)
        data: np.ndarray = adjacency.data.astype(np.float32)

        labels = labels.astype(np.int32)

        return fit_core(self.resolution, self.tol, node_probs_ou, node_probs_in,
                        self_loops, data, indices, indptr, labels)

    def _refine(self, labels, adjacency):
        refined_labels = np.arange(len(labels), dtype=int)
        origin = dict()
        values, counts = np.unique(labels, return_counts=True)
        counts = np.cumsum(counts)
        clusters = np.argsort(labels)
        singleton = np.ones(len(labels), dtype=bool)
        first_count, last_count = 0, counts[0]
        tot_weight = adjacency.sum()
        for value in values:
            cluster = clusters[first_count:last_count]
            sub_clusters = refined_labels[cluster]
            tot_cluster_weight = adjacency[cluster, :].sum()
            if value < values[-1]:
                first_count, last_count = counts[value], counts[value + 1]
            for i_global in cluster:
                node_weight = adjacency[i_global, :].sum()
                node_cluster = labels[i_global]
                if (adjacency[i_global, cluster].sum()
                    - adjacency[i_global, i_global]) >= self.resolution * node_weight \
                    * (tot_cluster_weight - node_weight):
                    if singleton[i_global]:
                        possibilities = dict()
                        node_mask = (refined_labels == node_cluster)
                        delta_out = adjacency[i_global, node_mask].sum() - self.resolution * \
                                        node_weight * adjacency[
                                        node_mask, node_mask].sum() / tot_weight
                        for sub_cluster in sub_clusters:
                            if sub_cluster != node_cluster:
                                global_mask = (refined_labels == sub_cluster)
                                sub_cluster_weight = adjacency[global_mask, global_mask].sum()
                                if adjacency[~global_mask, :][:, global_mask].sum() >= self.resolution \
                                    * sub_cluster_weight * (tot_cluster_weight - sub_cluster_weight):
                                    delta_in = adjacency[i_global, global_mask].sum() \
                                               - self.resolution * node_weight \
                                               * adjacency[global_mask, :][:, global_mask].sum() / \
                                               tot_weight
                                    if delta_in > delta_out:
                                        possibilities[sub_cluster] = (delta_in - delta_out) / self.random_factor
                        if possibilities:
                            probs = softmax(np.array(list(possibilities.values())))
                            refined_labels[i_global] = self.random_state.choice(np.array(list(possibilities.keys())),
                                                                                p=probs)
                            singleton[refined_labels == refined_labels[i_global]] = 0
            origin.update({sub_cluster: value for sub_cluster in set(refined_labels[cluster])})
        original_labels = [origin[sub_cluster] for sub_cluster in set(refined_labels)]
        _, refined_labels = np.unique(refined_labels, return_inverse=True)
        return refined_labels, np.array(original_labels)

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Leiden':
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`Louvain`
        """
        adjacency = check_format(adjacency)
        check_square(adjacency)
        n = adjacency.shape[0]
        labels_start = np.arange(n, dtype=int)

        if self.modularity == 'potts':
            probs_ou = check_probs('uniform', adjacency)
            probs_in = probs_ou.copy()
        elif self.modularity == 'newman':
            probs_ou = check_probs('degree', adjacency)
            probs_in = probs_ou.copy()
        elif self.modularity == 'dugue':
            probs_ou = check_probs('degree', adjacency)
            probs_in = check_probs('degree', adjacency.T)
        else:
            raise ValueError('Unknown modularity function.')

        nodes = np.arange(n, dtype=np.int32)
        if self.shuffle_nodes:
            nodes = self.random_state.permutation(nodes)
            adjacency = adjacency[nodes, :].tocsc()[:, nodes].tocsr()

        adjacency_clust = adjacency / adjacency.data.sum()

        membership = sparse.identity(n, format='csr')
        increase = True
        count_aggregations = 0
        self.log.print("Starting with", n, "nodes.")
        while increase:
            count_aggregations += 1

            labels_clust, pass_increase = self._optimize(adjacency_clust, probs_ou, probs_in, labels_start)
            _, labels_clust = np.unique(labels_clust, return_inverse=True)

            if pass_increase <= self.tol_aggregation:
                increase = False
            else:
                labels_refined, labels_start = self._refine(labels_clust, adjacency_clust)
                membership_refined = membership_matrix(labels_refined)
                membership = membership.dot(membership_refined)
                adjacency_clust, probs_ou, probs_in = self._aggregate(adjacency_clust, probs_ou, probs_in,
                                                                      membership_refined)

                n = adjacency_clust.shape[0]
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
        self._secondary_outputs(adjacency)

        return self
