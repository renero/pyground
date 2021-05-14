import pytest

from ..graph_utils import compute_graph_metrics


def test_compute_graph_metrics():
    dag1 = [('a', 'b'), ('a', 'c'), ('c', 'd'), ('c', 'b')]
    dag2 = [('a', 'b'), ('a', 'c'), ('b', 'd')]
    prec, rec = compute_graph_metrics(dag1, dag2)
    if prec != 0.75:
        pytest.fail('Wrong precision')
    if rec != 0.5:
        pytest.fail('Wrong precision')

    with pytest.raises(TypeError):
        compute_graph_metrics(0.0, 0.0)
