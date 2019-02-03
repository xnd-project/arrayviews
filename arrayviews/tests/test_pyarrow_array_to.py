import pytest
pa = pytest.importorskip("pyarrow")

try:
    import numpy as np
except ImportError:
    np = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import xnd
except ImportError:
    xnd = None

from arrayviews import pyarrow_array_to

numpytest=pytest.mark.skipif(
    np is None,
    reason="requires the numpy package")
pandastest=pytest.mark.skipif(
    pd is None,
    reason="requires the pandas package")
xndtest=pytest.mark.skipif(
    xnd is None,
    reason="requires the xnd package")

@numpytest
def test_numpy_ndarray():
    pa_arr = pa.array([1,2,3,4,5])
    arr = pyarrow_array_to.numpy_ndarray(pa_arr)
    expected_arr = np.array([1,2,3,4,5], dtype=np.int64)
    np.testing.assert_array_equal(arr, expected_arr)

@numpytest
def test_numpy_ndarray_with_null():
    pa_arr = pa.array([1,2,None,4,5])
    with pytest.raises(NotImplementedError, match="NumPy array view is only supported for arrays without nulls."):
        arr = pyarrow_array_to.numpy_ndarray(pa_arr)
        expected_arr = np.array([1,2,3,4,5], dtype=np.int64)
        np.testing.assert_array_equal(arr, expected_arr)

@pandastest
def test_pandas_series():
    pa_arr = pa.array([1,2,3,4,5])
    pd_ser = pyarrow_array_to.pandas_series(pa_arr)
    expected_pd_ser = pd.Series([1,2,3,4,5])
    assert (pd_ser == expected_pd_ser).all()

@pandastest
def test_pandas_series_with_null():
    pa_arr = pa.array([1,2,None,4,5])
    pd_ser = pyarrow_array_to.pandas_series(pa_arr)
    expected_pd_ser = pd.Series([1,2,None,4,5])
    assert (pd_ser.dropna().eq(expected_pd_ser.dropna())).all()

@xndtest
def test_xnd_xnd():
    pa_arr = pa.array([1,2,3,4,5])
    xnd_arr = pyarrow_array_to.xnd_xnd(pa_arr)
    expected_xnd_arr = xnd.xnd([1,2,3,4,5])
    assert xnd_arr.type == expected_xnd_arr.type
    assert xnd_arr.value == expected_xnd_arr.value

@xndtest
def test_xnd_xnd_with_null():
    pa_arr = pa.array([1,2,None,4,5])
    with pytest.raises(NotImplementedError, match="xnd view of pyarrow.Array with nulls"):
        xnd_arr = pyarrow_array_to.xnd_xnd(pa_arr)
        expected_xnd_arr = xnd.xnd([1,2,None,4,5])
        assert xnd_arr.type == expected_xnd_arr.type
        assert xnd_arr.value == expected_xnd_arr.value
