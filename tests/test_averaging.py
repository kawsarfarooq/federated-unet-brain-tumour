import numpy as np
import pytest
from brainseg_fl.averaging import average_updates


def test_updates_use_actual_example_counts():
    small = [np.array([[1., 2.]], dtype=np.float32)]
    large = [np.array([[5., 6.]], dtype=np.float32)]
    merged = average_updates([(small, 2), (large, 6)])
    np.testing.assert_allclose(merged[0], [[4., 5.]])
    assert merged[0].dtype == np.float32
    with pytest.raises(ValueError): average_updates([(small, 0)])
    with pytest.raises(ValueError): average_updates([(small, 1), ([np.ones(3)], 2)])
