import pytest

from ..utils import reset_seeds


#############################################################################
# The tests


def test_reset_seeds():
    try:
        reset_seeds(1)
    except AttributeError:
        pytest.fail('Unexpected error')
