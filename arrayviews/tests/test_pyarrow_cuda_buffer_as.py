import pytest
from arrayviews.cuda import numba_cuda_DeviceNDArray_as
from arrayviews.cuda import pyarrow_cuda_buffer_as
from arrayviews.cuda import cupy_ndarray_as


np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pa_cuda = pytest.importorskip("pyarrow.cuda")

try:
    import numba as nb
    import numba.cuda as nb_cuda
except ImportError:
    nb = nb_cuda = None

try:
    import cupy
except ImportError:
    cupy = None


numbatest = pytest.mark.skipif(
    nb is None or nb_cuda is None,
    reason="requires the numba and numba.cuda packages")

cupytest = pytest.mark.skipif(
    cupy is None,
    reason="requires the cupy package")


@numbatest
def test_numba_cuda_DeviceNDArray():
    pa_cbuf = pyarrow_cuda_buffer_as.random(5)
    nb_arr = pyarrow_cuda_buffer_as.numba_cuda_DeviceNDArray(pa_cbuf)

    arr1 = numba_cuda_DeviceNDArray_as.numpy_ndarray(nb_arr)
    arr2 = pyarrow_cuda_buffer_as.numpy_ndarray(pa_cbuf)
    np.testing.assert_array_equal(arr1, arr2)

    nb_arr[1] = 99
    arr1 = numba_cuda_DeviceNDArray_as.numpy_ndarray(nb_arr)
    arr2 = pyarrow_cuda_buffer_as.numpy_ndarray(pa_cbuf)
    np.testing.assert_array_equal(arr1, arr2)
    assert arr1[1] == 99


@cupytest
def test_cupy_ndarray():
    pa_cbuf = pyarrow_cuda_buffer_as.random(5)
    cp_arr = pyarrow_cuda_buffer_as.cupy_ndarray(pa_cbuf)

    arr1 = cupy_ndarray_as.numpy_ndarray(cp_arr)
    arr2 = pyarrow_cuda_buffer_as.numpy_ndarray(pa_cbuf)
    np.testing.assert_array_equal(arr1, arr2)

    cp_arr[1] = 99
    arr1 = cupy_ndarray_as.numpy_ndarray(cp_arr)
    arr2 = pyarrow_cuda_buffer_as.numpy_ndarray(pa_cbuf)
    np.testing.assert_array_equal(arr1, arr2)
    assert arr1[1] == 99
