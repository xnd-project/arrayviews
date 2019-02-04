import pytest
from arrayviews import pandas_series_as
pd = pytest.importorskip("pandas")

try:
    import numpy as np
except ImportError:
    np = None
try:
    import pyarrow as pa
except ImportError:
    pa = None
try:
    import xnd
except ImportError:
    xnd = None

numpytest = pytest.mark.skipif(
    np is None,
    reason="requires the numpy package")
pyarrowtest = pytest.mark.skipif(
    pa is None,
    reason="requires the pyarrow package")
xndtest = pytest.mark.skipif(
    xnd is None,
    reason="requires the xnd package")


@numpytest
def test_numpy_ndarray():
    pd_ser = pd.Series([1, 2, 3, 4, 5])
    arr = pandas_series_as.numpy_ndarray(pd_ser)
    expected_arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    np.testing.assert_array_equal(arr, expected_arr)

    pd_ser[1] = 999
    assert arr[1] == 999


@numpytest
def test_numpy_ndarray_with_null():
    pd_ser = pd.Series([1, 2, None, 4, 5], dtype=np.float64)
    arr = pandas_series_as.numpy_ndarray(pd_ser)
    expected_arr = np.array([1, 2, np.nan, 4, 5], dtype=np.float64)
    np.testing.assert_array_equal(arr, expected_arr)

    pd_ser[1] = 999
    assert arr[1] == 999


@pyarrowtest
def test_pyarrow_array():
    pd_ser = pd.Series([1, 2, 3, 4, 5])
    pa_arr = pandas_series_as.pyarrow_array(pd_ser)
    expected_pa_arr = pa.array([1, 2, 3, 4, 5])
    assert pa_arr.to_pylist() == expected_pa_arr.to_pylist()

    pd_ser[1] = 999
    assert pa_arr[1] == 999


@pyarrowtest
def test_pyarrow_array_with_null():
    pd_ser = pd.Series([1, 2, None, 4, 5])
    expected_pa_arr = pa.array([1, 2, None, 4, 5], type=pa.float64())
    pa_arr = pandas_series_as.pyarrow_array(pd_ser, nan_to_null=True)
    assert pa_arr.null_count == expected_pa_arr.null_count
    assert pa_arr.to_pylist() == expected_pa_arr.to_pylist()

    pd_ser[1] = 999
    assert pa_arr[1] == 999

    pd_ser = pd.Series([1, 2, None, 4, 5]*5)
    expected_pa_arr = pa.array([1, 2, None, 4, 5]*5, type=pa.float64())
    pa_arr = pandas_series_as.pyarrow_array(pd_ser, nan_to_null=True)
    assert pa_arr.null_count == expected_pa_arr.null_count
    assert pa_arr.to_pylist() == expected_pa_arr.to_pylist()


@xndtest
def test_xnd_xnd():
    pd_ser = pd.Series([1, 2, 3, 4, 5])
    xnd_arr = pandas_series_as.xnd_xnd(pd_ser)
    expected_xnd_arr = xnd.xnd([1, 2, 3, 4, 5])
    assert xnd_arr.type == expected_xnd_arr.type
    assert xnd_arr.value == expected_xnd_arr.value

    pd_ser[1] = 999
    assert xnd_arr[1] == 999


@xndtest
def test_xnd_xnd_with_null():
    pd_ser = pd.Series([1, 2, None, 4, 5], dtype=np.float64)
    with pytest.raises(NotImplementedError,
                       match="xnd view of pandas.Series with nans"):
        xnd_arr = pandas_series_as.xnd_xnd(pd_ser, nan_to_null=True)
        expected_xnd_arr = xnd.xnd([1, 2, None, 4, 5], dtype='?float64')
        assert xnd_arr.type == expected_xnd_arr.type
        assert xnd_arr.value == expected_xnd_arr.value

        pd_ser[1] = 999
        assert xnd_arr[1] == 999
