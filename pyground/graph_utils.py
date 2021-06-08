"""
This module incorporates util functions for graphs.
"""
from pathlib import Path
from typing import List, Union

import networkx as nx
import numpy as np
import pandas as pd
import pydot as pydot
import pydotplus
from networkx import Graph, DiGraph

from pyground.file_utils import file_exists


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

    # Check if this graph contain weight information
    get_edges = getattr(graph, "edges", None)
    if callable(get_edges):
        edges_weights = get_edges(data='weight')
    else:
        edges_weights = edges

    # Printout
    for edge in edges_weights:
        if len(edge) == 3:
            print(("{:" + str(mx) + "s} –– {:" + str(mx) + "s} {:+.4f}").format(
                edge[0], edge[1], edge[2]))
        else:
            print(("{:" + str(mx) + "s} –– {:" + str(mx) + "s}").format(
                edge[0], edge[1]))


def graph_from_adjacency(adjacency: np.ndarray, node_labels=None) -> nx.DiGraph:
    """
    Manually parse the adj matrix to shape a dot graph

    Args:
        adjacency: a numpy adjacency matrix
        node_labels: an array of same length as nr of columns in the adjacency
        matrix containing the labels to use with every node.

    Returns:
         The Graph (DiGraph)

    """
    G = nx.DiGraph()
    G.add_nodes_from(range(adjacency.shape[1]))
    arrowhead = ["none", "odot", "normal"]
    for i, row in enumerate(adjacency):
        for j, value in enumerate(row):
            if value != 0:
                G.add_edge(i, j, arrowhead=arrowhead[value])

    # Map the current column numbers to the letters used in toy dataset
    if node_labels is not None and len(node_labels) == adjacency.shape[1]:
        mapping = dict(zip(sorted(G), node_labels))
        G = nx.relabel_nodes(G, mapping)

    return G


def graph_from_dot(dot_object: pydot.Dot) -> nx.DiGraph:
    """ Returns a NetworkX DiGraph from a DOT object. """
    dotplus = pydotplus.graph_from_dot_data(dot_object.to_string())
    dotplus.set_strict(True)
    return nx.nx_pydot.from_pydot(dotplus)


def read_graph_fom_csv(output_path: str,
                        graph_file: str,
                        graph_type: Union[Graph, DiGraph]):
    """
    Read Graph from a CSV file with "FROM", "TO" and "WEIGHT" fields
    """
    edges = pd.read_csv(Path(output_path) / graph_file)
    Graphtype = graph_type()
    ugraph = nx.from_pandas_edgelist(edges,
                                     source='from',
                                     target='to',
                                     edge_attr='weight',
                                     create_using=Graphtype)
    return ugraph


def save_graph_to_csv(graph, output_file):
    """
    Save a GrAPH to CSV file with "FROM", "TO" and "CSV"
    """
    if file_exists(output_file, "."):
        output_file = f"New_{output_file}"
    skeleton = pd.DataFrame(list(graph.edges(data='weight')))
    skeleton.columns = ['from', 'to', 'weight']
    skeleton.to_csv(output_file, index=False)
