import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pydot
from IPython.display import Image, display
from scipy.cluster import hierarchy

from pyground.graph_utils import graph_to_adjacency


def dot_graph(G: nx.DiGraph, undirected=False, plot: bool = True, **kwargs) -> None:
    """
    Display a DOT of the graph in the notebook.

    Args:
        G (nx.Graph or DiGraph): the graph to be represented.
        undirected (bool): default False, indicates whether the plot is forced
            to contain no arrows.
        plot (bool): default is True, this flag can be used to simply generate
            the object but not plot, in case the object is needed to generate
            a PNG version of the DOT, for instance.

    Returns:
        pydot.Dot object
    """
    if len(list(G.edges())) == 0:
        return None
    # Obtain the DOT version of the NX.DiGraph and visualize it.
    G = G.to_undirected() if undirected else G
    dot_object = nx.nx_pydot.to_pydot(G)

    # This is to display single arrows with two heads instead of two arrows with
    # one head towards each direction.
    dot_object.set_concentrate(True)
    if plot:
        plot_dot(dot_object, **kwargs)

    return dot_object


def plot_dot(dot_object: pydot.Dot, **kwargs) -> None:
    """ Displays a DOT object in the notebook """
    image = Image(dot_object.create_png(), **kwargs)
    display(image)


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
            edge_colors[edge] = 'orange'

    # set node colors
    G_nodes = set(G.nodes())
    H_nodes = set(H.nodes())
    node_colors = []
    for node in GH.nodes():
        if node in G_nodes:
            if node in H_nodes:
                node_colors.append('lightgrey')
                continue
            node_colors.append('lightgreen')
        if node in H_nodes:
            node_colors.append('orange')

    pos = nx.circular_layout(GH, scale=20)
    nx.draw(GH, pos,
            nodelist=GH.nodes(),
            node_color=node_colors,
            edgelist=edge_colors.keys(),
            edge_color=edge_colors.values(),
            node_size=800,
            width=2, alpha=0.5,
            with_labels=True)


def plot_adjacency(g: nx.Graph, ax = None):
    """
    Plots the adjacency matrix as explained by scikit contributor
    Andreas Mueller in Columbia lectures, ordering and grouping
    (numerical) features with higher correlation.

    Returns:
        None
    """
    mat = graph_to_adjacency(g)
    features = sorted(list(g.nodes))
    num_features = len(features)

    if ax is None:
        _, ax = plt.subplots()
    plt.xticks(fontsize=10)
    ax.set_title("Grouped Adjacency Matrix")
    ax.matshow(mat, interpolation="nearest")
    for (j, i), label in np.ndenumerate(mat):
        ax.text(i, j, f"{label:.2g}", ha="center", va="center")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks(range(num_features))
    ax.set_xticklabels(features)
    ax.set_yticks(range(num_features))
    ax.set_yticklabels(features)
