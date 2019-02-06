import pytest
from arrayviews.cuda import numba_cuda_DeviceNDArray_as
from arrayviews.cuda import cudf_Series_as

cf = pytest.importorskip("cudf")
np = pytest.importorskip("numpy")

try:
    import numba as nb
    import numba.cuda as nb_cuda
except ImportError:
    nb = nb_cuda = None

numbatest = pytest.mark.skipif(
    nb is None or nb_cuda is None,
    reason="requires the numba and numba.cuda packages")


@numbatest
def test_numba_cuda_DeviceNDArray():
    cf_ser = cudf_Series_as.random(5)
    nb_arr = cudf_Series_as.numba_cuda_DeviceNDArray(cf_ser)

    arr1 = numba_cuda_DeviceNDArray_as.numpy_ndarray(nb_arr)
    arr2 = cudf_Series_as.numpy_ndarray(cf_ser)
    np.testing.assert_array_equal(arr1, arr2)

    nb_arr[1] = 99
    arr1 = numba_cuda_DeviceNDArray_as.numpy_ndarray(nb_arr)
    arr2 = cudf_Series_as.numpy_ndarray(cf_ser)
    np.testing.assert_array_equal(arr1, arr2)
    assert arr1[1] == 99
