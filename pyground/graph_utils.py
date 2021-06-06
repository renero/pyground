"""
This module incorporates util functions for graphs.
"""
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydot as pydot
import pydotplus
from IPython.display import Image, display


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


def dot_graph(G: nx.DiGraph) -> None:
    """
    Display a DOT of the graph in the notebook.
    """
    # Obtain the DOT version of the NX.DiGraph and visualize it.
    dot_graph = nx.nx_pydot.to_pydot(G)

    # This is to display single arrows with two heads instead of two arrows with
    # one head towards each direction.
    dot_graph.set_concentrate(True)
    plot_dot(dot_graph)


def plot_dot(pdot: pydot.Dot) -> None:
    """ Displays a DOT object in the notebook """
    plt = Image(pdot.create_png())
    display(plt)


def plot_graph(graph: nx.DiGraph) -> None:
    """Plot a graph using default Matplotlib methods"""
    pos = nx.circular_layout(graph, scale=20)
    nx.draw(graph, pos,
            nodelist=graph.nodes(),
            node_color="lightblue",
            node_size=800,
            width=2,
            alpha=0.9,
            with_labels=True)
    plt.show()


def plot_graphs(G: nx.MultiDiGraph, H: nx.DiGraph) -> None:
    """Plot two graphs side by side."""
    pos1 = nx.circular_layout(G, scale=20)
    pos2 = nx.circular_layout(H, scale=20)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax = axes.flatten()
    nx.draw_networkx(G, pos1, node_color="lightblue",
                     node_size=800, edge_color='orange',
                     width=2, alpha=0.9, ax=ax[0])
    ax[0].set_axis_off()
    ax[0].set_title("Ground Truth")
    nx.draw_networkx(H, pos2, node_color="lightblue",
                     node_size=800, edge_color='lightblue',
                     width=2, alpha=0.9, ax=ax[1])
    ax[1].set_axis_off()
    ax[1].set_title("Other")
    plt.tight_layout()
    plt.show()


def plot_compared_graph(G: nx.DiGraph, H: nx.DiGraph) -> None:
    """
    Iterate over the composed graph's edges and nodes, and assign to these a
    color depending on which graph they belong to (including both at the same
    time too). This could also be extended to adding some attribute indicating
    to which graph it belongs too.
    Intersecting nodes and edges will have a magenta color. Otherwise they'll
    be green or blue if they belong to the G or H Graph respectively
    """
    GH = nx.compose(G, H)
    # set edge colors
    edge_colors = dict()
    for edge in GH.edges():
        if G.has_edge(*edge):
            if H.has_edge(*edge):
                edge_colors[edge] = 'black'
                continue
            edge_colors[edge] = 'lightgreen'
        elif H.has_edge(*edge):
            edge_colors[edge] = 'lightblue'

    # set node colors
    G_nodes = set(G.nodes())
    H_nodes = set(H.nodes())
    node_colors = []
    for node in GH.nodes():
        if node in G_nodes:
            if node in H_nodes:
                node_colors.append('green')
                continue
            node_colors.append('lightgreen')
        if node in H_nodes:
            node_colors.append('lightblue')

    pos = nx.circular_layout(GH, scale=20)
    nx.draw(GH, pos,
            nodelist=GH.nodes(),
            node_color=node_colors,
            edgelist=edge_colors.keys(),
            edge_color=edge_colors.values(),
            node_size=800,
            width=2, alpha=0.5,
            with_labels=True)
