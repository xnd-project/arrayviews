
import pytest
from arrayviews import xnd_xnd_as
xnd = pytest.importorskip("xnd")

try:
    import pyarrow as pa
except ImportError:
    pa = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import numpy as np
except ImportError:
    np = None


pyarrowtest = pytest.mark.skipif(
    pa is None,
    reason="requires the pyarrow package")
pandastest = pytest.mark.skipif(
    pd is None,
    reason="requires the pandas package")
numpytest = pytest.mark.skipif(
    np is None,
    reason="requires the numpy package")


@pyarrowtest
def test_pyarrow_array():
    xd_arr = xnd.xnd([1, 2, 3, 4, 5])
    pa_arr = xnd_xnd_as.pyarrow_array(xd_arr)
    expected_pa_arr = pa.array([1, 2, 3, 4, 5])
    assert pa_arr.to_pylist() == expected_pa_arr.to_pylist()
    xd_arr[1] = 999
    assert pa_arr[1] == 999


@pyarrowtest
def test_pyarrow_array_with_null():
    xd_arr = xnd.xnd([1, 2, None, 4, 5])
    with pytest.raises(
            NotImplementedError,
            match="pyarrow.Array view of xnd.xnd with optional values"):
        pa_arr = xnd_xnd_as.pyarrow_array(xd_arr)
        expected_pa_arr = pa.array([1, 2, None, 4, 5], type=pa.float64())
        assert pa_arr.to_pylist() == expected_pa_arr.to_pylist()

        xd_arr[1] = 999
        assert pa_arr[1] == 999


@pandastest
def test_pandas_series():
    xd_arr = xnd.xnd([1, 2, 3, 4, 5])
    pd_ser = xnd_xnd_as.pandas_series(xd_arr)
    expected_pd_ser = pd.Series([1, 2, 3, 4, 5])
    assert (pd_ser == expected_pd_ser).all()

    xd_arr[1] = 999
    assert pd_ser[1] == 999


@pandastest
def test_pandas_series_with_null():
    xd_arr = xnd.xnd([1, 2, None, 4, 5])
    with pytest.raises(
            NotImplementedError,
            match="pandas.Series view of xnd.xnd with optional values"):
        pd_ser = xnd_xnd_as.pandas_series(xd_arr)
        expected_pd_ser = pd.Series([1, 2, None, 4, 5])
        assert (pd_ser.dropna().eq(expected_pd_ser.dropna())).all()

        xd_arr[1] = 999
        assert pd_ser[1] == 999


@numpytest
def test_numpy_ndarray():
    xd_arr = xnd.xnd([1, 2, 3, 4, 5])
    np_arr = xnd_xnd_as.numpy_ndarray(xd_arr)
    expected_np_arr = np.array([1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np_arr, expected_np_arr)

    xd_arr[1] = 999
    assert np_arr[1] == 999


@numpytest
def test_numpy_ndarray_with_null():
    xd_arr = xnd.xnd([1, 2, None, 4, 5])
    with pytest.raises(
            NotImplementedError,
            match="numpy.ndarray view of xnd.xnd with optional values"):
        np_arr = xnd_xnd_as.numpy_ndarray(xd_arr)
        expected_np_arr = np.array([1, 2, np.nan, 4, 5])
        np.testing.assert_array_equal(np_arr, expected_np_arr)

        xd_arr[1] = 999
        assert np_arr[1] == 999
