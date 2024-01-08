import numpy as np
from sklearn.metrics.cluster import pair_confusion_matrix

def pairwise_f1(y_true, y_pred):
    C = pair_confusion_matrix(y_true, y_pred)
    p = C[1][1] / (C[1][1] + C[0][1])
    r = C[1][1] / (C[1][1] + C[1][0])
    
    return (2 * p * r) / (p + r)

def pairwise_acc(y_true, y_pred):
    C = pair_confusion_matrix(y_true, y_pred)
    
    return (C[1][1] + C[0][1]) / (C[1][1] + C[0][1] + C[1][0] + C[0][0])

def conductance(adjacency, clusters):
    """Computes graph conductance as in Yang & Leskovec (2012); conductance of 1 cluster is the fraction of total edge volume that points outside the cluster.

    Args:
        adjacency: Input graph in terms of its sparse adjacency matrix.
        clusters: An (n,) int cluster vector.

    Returns:
        The average conductance value of the graph clusters.
    """
    inter = 0  # Number of inter-cluster edges.
    intra = 0  # Number of intra-cluster edges.
    cluster_indices = np.zeros(adjacency.size(0), dtype=bool)
    for cluster_id in np.unique(clusters):
        cluster_indices[:] = 0
        cluster_indices[np.where(clusters == cluster_id)[0]] = 1
        adj_submatrix = adjacency[cluster_indices, :].to_dense().numpy()
        inter += np.sum(adj_submatrix[:, cluster_indices])
        intra += np.sum(adj_submatrix[:, ~cluster_indices])
    return intra / (inter + intra)

def modularity(adjacency, clusters):
    """Computes graph modularity; the fraction of the edges that fall within a given cluster minus the expected fraction if edges were distributed at random (i.e. in a random graph with identical degree sequence).

    Args:
        adjacency: Input graph in terms of its sparse adjacency matrix.
        clusters: An (n,) int cluster vector.

    Returns:
        The value of graph modularity.
        https://en.wikipedia.org/wiki/Modularity_(networks)
    """
    degrees = adjacency.sum(dim=0).numpy()
    n_edges = degrees.sum()  # Note that it's actually 2*n_edges.
    result = 0
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices].to_dense().numpy()
        degrees_submatrix = degrees[cluster_indices]
        result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix)**2) / n_edges
    return result / n_edges