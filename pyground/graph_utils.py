"""
This module incorporates util functions for graphs.
"""
from typing import List

import networkx as nx
import numpy as np


def compute_graph_metrics(truth, result):
    """
    Compute graph precision and recall. Recall refers to the list of edges
    that have been correctly identified in result, and precision, to the
    ratio of edges that correctly math to those in the ground truth.

    Arguments:
        truth: A list of edges representing the true structure of the graph
               to compare with.
        result: The dag for which to measure the metrics.

    Returns:
        precision, recall values as floats

    Example:
        >>> dag1 = [('a', 'b'), ('a', 'c'), ('c', 'd'), ('c', 'b')]
        >>> dag2 = [('a', 'b'), ('a', 'c'), ('b', 'd')]
        >>> prec, rec = compute_graph_metrics(dag1, dag2)
        >>> print(prec, rec)
        >>> 0.75 0.5

    """
    # Convert the ground truth and target into a set of tuples with edges
    if not isinstance(truth, set):
        ground_truth = set([tuple(pair) for pair in truth])
    elif isinstance(truth, set):
        ground_truth = truth
    else:
        raise TypeError("Truth argument must be a list or a set.")
    if not isinstance(result, set):
        target = set([tuple(pair) for pair in result])
    elif isinstance(result, set):
        target = result
    else:
        raise TypeError("Results argument must be a list or a set.")

    # Set the total number of edges if ground truth skeleton
    total = float(len(ground_truth))
    true_positives = len(ground_truth.intersection(target))
    false_positives = len(target - ground_truth.intersection(target))
    precision = 1. - (false_positives / total)
    recall = true_positives / total

    return precision, recall


def build_graph(list_nodes: List, matrix: np.ndarray,
                threshold=0.05, zero_diag=True) -> nx.Graph:
    """
    Builds a graph from an adjacency matrix. For each position i, j, if the
    value is greater than the threshold, an edge is added to the graph. The
    names of the vertices are in the list of nodes pased as argument, whose
    order must match the columns in the matrix.

    The diagonal of the matrix is set to zero to avoid inner edges, but this
    behavior can be overridden by setting zero_diag to False.

    Args:
        list_nodes: a list with the names of the graph's nodes.
        matrix: a numpy ndarray with the weights to be used
        threshold: the threshold above which a vertex is created in the graph
        zero_diag: boolean indicating whether zeroing the diagonal. Def True.

    Returns:
        nx.Graph: A graph with edges between values > threshold.

    Example:
        >>> matrix = np.array([[0., 0.3, 0.2],[0.3, 0., 0.2], [0.0, 0.2, 0.]])
        >>> dag = build_graph(['a','b','c'], matrix, threshold=0.1)
        >>> dag.edges()
            EdgeView([('a', 'b'), ('a', 'c'), ('b', 'c')])
    """
    M = np.copy(matrix)
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")
    if M.shape[1] != len(list_nodes):
        raise ValueError("List of nodes doesn't match number of rows/cols")
    if zero_diag:
        np.fill_diagonal(M, 0.)
    graph = nx.Graph()
    for (i, j), x in np.ndenumerate(M):
        if M[i, j] > threshold:
            graph.add_edge(list_nodes[i], list_nodes[j],
                           weight=M[i, j])
    for node in list_nodes:
        if node not in graph.nodes():
            graph.add_node(node)
    return graph


def print_graph_edges(graph: nx.Graph):
    """
    Pretty print the nodes of a graph, with weights

    Args:
         graph: the graph to be printed out.
    Returns:
        None.
    Example:
        >>> matrix = np.array([[0., 0.3, 0.2],[0.3, 0., 0.2], [0.0, 0.2, 0.]])
        >>> dag = build_graph(['a','b','c'], matrix, threshold=0.1)
        >>> print_graph_edges(dag)
            Graph contains 3 edges.
            a –– b +0.3000
            a –– c +0.2000
            b –– c +0.2000

    """
    mx = max([len(s) for s in list(graph.nodes)])
    edges = list(graph.edges)
    print(f'Graph contains {len(edges)} edges.')
    for edge in graph.edges(data='weight'):
        print(("{:" + str(mx) + "s} –– {:" + str(mx) + "s} {:+.4f}").format(
            edge[0], edge[1], edge[2]))
