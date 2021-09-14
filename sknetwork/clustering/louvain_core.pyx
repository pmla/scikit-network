# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
from libc.stdlib cimport rand, srand
from libcpp.set cimport set
from libcpp.vector cimport vector
from libcpp.queue cimport queue
cimport cython

ctypedef fused int_or_long:
    int
    long

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline int_or_long randint(int_or_long lower, int_or_long upper) nogil:
    if upper == lower:
        return lower
    else:
        return ( rand() % (upper - lower) ) + lower

@cython.boundscheck(False)
@cython.wraparound(False)
def fit_core(float resolution, float tol, float[:] ou_node_probs, float[:] in_node_probs, float[:] ou_cluster_probs,
             float[:] in_cluster_probs, float[:] self_loops, float[:] data, int_or_long[:] indices,
             int_or_long[:] indptr, int_or_long[:] labels_array, bint random_move, bint fast_move, int_or_long seed):  # pragma: no cover
    """Fit the clusters to the objective function.

    Parameters
    ----------
    resolution :
        Resolution parameter (positive).
    tol :
        Minimum increase in modularity to enter a new optimization pass.
    ou_node_probs :
        Distribution of node weights based on their out-edges (sums to 1).
    in_node_probs :
        Distribution of node weights based on their in-edges (sums to 1).
    ou_cluster_probs:
        Initial out-degrees of the clusters
    in_cluster_probs:
        Initial in-degrees of the clusters
    self_loops :
        Weights of self loops.
    data :
        CSR format data array of the normalized adjacency matrix.
    indices :
        CSR format index array of the normalized adjacency matrix.
    indptr :
        CSR format index pointer array of the normalized adjacency matrix.
    labels_array:
        Pre-existing labels of the nodes
    random_move :
        Enables a random neighbor candidate to be picked rather than looking at the whole neighborhood
    fast_move :
        Enables iterations on nodes whose neighborhood has been changed only
    seed :
        Random seed to be used (inferred from the random_state of the Louvain instance)

    Returns
    -------
    labels :
        Cluster index of each node.
    total_increase :
        Score of the clustering (total increase in modularity).
    """
    cdef int_or_long n = indptr.shape[0] - 1
    cdef int_or_long n_clusters = in_node_probs.shape[0]
    cdef int_or_long increase = 1
    cdef int_or_long cluster
    cdef int_or_long cluster_best
    cdef int_or_long cluster_node
    cdef int_or_long i
    cdef int_or_long j
    cdef int_or_long j1
    cdef int_or_long j2
    cdef int_or_long label

    cdef float increase_total = 0
    cdef float increase_pass
    cdef float delta
    cdef float delta_best
    cdef float delta_exit
    cdef float delta_local
    cdef float node_prob_in
    cdef float node_prob_ou
    cdef float ratio_in
    cdef float ratio_ou

    cdef queue[int_or_long] nodes
    cdef queue[int_or_long] next_nodes
    cdef vector[bint] added_to_queue
    cdef vector[int_or_long] labels
    cdef vector[float] neighbor_clusters_weights
    cdef vector[float] ou_clusters_weights
    cdef vector[float] in_clusters_weights
    cdef set[int_or_long] unique_clusters = ()

    srand(seed)

    for i in range(n_clusters):
        neighbor_clusters_weights.push_back(0.)
        ou_clusters_weights.push_back(ou_cluster_probs[i])
        in_clusters_weights.push_back(in_cluster_probs[i])

    for i in range(n):
        next_nodes.push(i)
        added_to_queue.push_back(1)
        labels.push_back(labels_array[i])

    while increase == 1 and not next_nodes.empty():
        increase = 0
        increase_pass = 0

        nodes = next_nodes
        next_nodes.swap(queue[int_or_long]())

        while not nodes.empty():

            i = nodes.front()
            nodes.pop()
            if fast_move:
                added_to_queue[i] = 0
            else:
                next_nodes.push(i)
            cluster_node = labels[i]
            j1 = indptr[i]
            j2 = indptr[i + 1]

            if random_move:

                j = indices[randint(j1, j2)]
                cluster = labels[j]

                if cluster != cluster_node:

                    for j in range(j1, j2):
                        if labels[indices[j]] == cluster:
                            neighbor_clusters_weights[cluster] += data[j]
                        if labels[indices[j]] == cluster_node:
                            neighbor_clusters_weights[cluster_node] += data[j]

                    node_prob_ou = ou_node_probs[i]
                    node_prob_in = in_node_probs[i]
                    ratio_ou = resolution * node_prob_ou
                    ratio_in = resolution * node_prob_in

                    delta_exit = 2 * (neighbor_clusters_weights[cluster_node] - self_loops[i])
                    delta_exit -= ratio_ou * (in_clusters_weights[cluster_node] - node_prob_in)
                    delta_exit -= ratio_in * (ou_clusters_weights[cluster_node] - node_prob_ou)

                    delta = 2 * neighbor_clusters_weights[cluster]
                    delta -= ratio_ou * in_clusters_weights[cluster]
                    delta -= ratio_in * ou_clusters_weights[cluster]

                    delta_local = delta - delta_exit

                    if delta_local > 0:
                        increase_pass += delta_local
                        ou_clusters_weights[cluster_node] -= node_prob_ou
                        in_clusters_weights[cluster_node] -= node_prob_in
                        ou_clusters_weights[cluster] += node_prob_ou
                        in_clusters_weights[cluster] += node_prob_in
                        labels[i] = cluster
                        if fast_move:
                            for j in range(j1, j2):
                                if labels[indices[j]] != cluster and added_to_queue[indices[j]] == 0:
                                    added_to_queue[indices[j]] = 1
                                    next_nodes.push(indices[j])

                    neighbor_clusters_weights[cluster] = 0

            else:

                for j in range(j1, j2):
                    label = labels[indices[j]]
                    neighbor_clusters_weights[label] += data[j]
                    unique_clusters.insert(label)

                unique_clusters.erase(cluster_node)

                if not unique_clusters.empty():
                    node_prob_ou = ou_node_probs[i]
                    node_prob_in = in_node_probs[i]
                    ratio_ou = resolution * node_prob_ou
                    ratio_in = resolution * node_prob_in

                    delta_exit = 2 * (neighbor_clusters_weights[cluster_node] - self_loops[i])
                    delta_exit -= ratio_ou * (in_clusters_weights[cluster_node] - node_prob_in)
                    delta_exit -= ratio_in * (ou_clusters_weights[cluster_node] - node_prob_ou)

                    delta_best = 0
                    cluster_best = cluster_node

                    for cluster in unique_clusters:
                        delta = 2 * neighbor_clusters_weights[cluster]
                        delta -= ratio_ou * in_clusters_weights[cluster]
                        delta -= ratio_in * ou_clusters_weights[cluster]

                        delta_local = delta - delta_exit
                        if delta_local > delta_best:
                            delta_best = delta_local
                            cluster_best = cluster

                        neighbor_clusters_weights[cluster] = 0

                    if delta_best > 0:
                        increase_pass += delta_best
                        ou_clusters_weights[cluster_node] -= node_prob_ou
                        in_clusters_weights[cluster_node] -= node_prob_in
                        ou_clusters_weights[cluster_best] += node_prob_ou
                        in_clusters_weights[cluster_best] += node_prob_in
                        labels[i] = cluster_best
                        if fast_move:
                            for j in range(j1, j2):
                                if labels[indices[j]] != cluster and added_to_queue[indices[j]] == 0:
                                    added_to_queue[indices[j]] = 1
                                    next_nodes.push(indices[j])

                    unique_clusters.clear()

            neighbor_clusters_weights[cluster_node] = 0

        increase_total += increase_pass
        if increase_pass > tol:
            increase = 1
    return labels, increase_total
