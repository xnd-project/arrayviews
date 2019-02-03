
import pytest

np = pytest.importorskip("numpy")

try:
    import pyarrow as pa
except ImportError:
    pa = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import xnd
except ImportError:
    xnd = None

from arrayviews import numpy_ndarray_to

pyarrowtest=pytest.mark.skipif(
    pa is None,
    reason="requires the pyarrow package")
pandastest=pytest.mark.skipif(
    pd is None,
    reason="requires the pandas package")
xndtest=pytest.mark.skipif(
    xnd is None,
    reason="requires the xnd package")


@pyarrowtest
def test_pyarrow_array():
    arr = np.array([1,2,3,4,5])
    pa_arr = numpy_ndarray_to.pyarrow_array(arr)
    expected_pa_arr = pa.array([1,2,3,4,5])
    assert pa_arr.to_pylist() == expected_pa_arr.to_pylist()

@pyarrowtest
def test_pyarrow_array_with_null():
    arr = np.array([1,2,np.nan,4,5])
    pa_arr = numpy_ndarray_to.pyarrow_array(arr)
    expected_pa_arr = pa.array([1,2,None,4,5], type=pa.float64())
    assert pa_arr.to_pylist() == expected_pa_arr.to_pylist()

    arr = np.array([1,2,np.nan,4,5]*5)
    pa_arr = numpy_ndarray_to.pyarrow_array(arr)
    expected_pa_arr = pa.array([1,2,None,4,5]*5, type=pa.float64())
    assert pa_arr.to_pylist() == expected_pa_arr.to_pylist()

@pandastest
def test_pandas_series():
    arr = np.array([1,2,3,4,5])
    pd_ser = numpy_ndarray_to.pandas_series(arr)
    expected_pd_ser = pd.Series([1,2,3,4,5])
    assert (pd_ser == expected_pd_ser).all()

@pandastest
def test_pandas_series_with_null():
    arr = np.array([1,2,np.nan,4,5])
    pd_ser = numpy_ndarray_to.pandas_series(arr)
    expected_pd_ser = pd.Series([1,2,None,4,5])
    assert (pd_ser.dropna().eq(expected_pd_ser.dropna())).all()

@xndtest
def test_xnd():
    arr = np.array([1,2,3,4,5])
    xnd_arr = numpy_ndarray_to.xnd_xnd(arr)
    expected_xnd_arr = xnd.xnd([1,2,3,4,5])
    assert xnd_arr.type == expected_xnd_arr.type
    assert xnd_arr.value == expected_xnd_arr.value

@xndtest
def test_xnd_with_null():
    arr = np.array([1,2,np.nan,4,5])
    with pytest.raises(NotImplementedError, match="xnd view of numpy ndarray with nans"):
        xnd_arr = numpy_ndarray_to.xnd_xnd(arr)
        expected_xnd_arr = xnd.xnd([1,2,None,4,5], type = '5 * ?float64')
        assert xnd_arr.type == expected_xnd_arr.type
        assert xnd_arr.value == expected_xnd_arr.value
