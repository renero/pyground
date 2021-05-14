"""
This module incorporates util functions for graphs.
"""


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
