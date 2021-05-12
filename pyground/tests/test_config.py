import pytest

from ..config import config


#############################################################################
# The tests

def test_config():
    params = config("kk")
    assert len(params) >= 2, pytest.fail("No Configuration loaded")
