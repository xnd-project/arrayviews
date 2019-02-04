
import pytest
from arrayviews.cuda import numba_cuda_DeviceNDArray_as
from arrayviews.cuda import pyarrow_cuda_buffer_as

np = pytest.importorskip("numpy")
nb = pytest.importorskip("numba")
nb_cuda = pytest.importorskip("numba.cuda")

try:
    import pyarrow as pa
    import pyarrow.cuda as pa_cuda
except ImportError:
    pa = pa_cuda = None

pyarrowtest = pytest.mark.skipif(
    pa is None or pa_cuda is None,
    reason="requires the pyarrow and pyarrow.cuda packages")


@pyarrowtest
def test_pyarrow_cuda_buffer():
    nb_arr = numba_cuda_DeviceNDArray_as.random(5)
    pa_cbuf = numba_cuda_DeviceNDArray_as.pyarrow_cuda_buffer(nb_arr)
    arr1 = numba_cuda_DeviceNDArray_as.numpy_ndarray(nb_arr)
    arr2 = pyarrow_cuda_buffer_as.numpy_ndarray(pa_cbuf)
    np.testing.assert_array_equal(arr1, arr2)

    nb_arr[1] = 99
    arr1 = numba_cuda_DeviceNDArray_as.numpy_ndarray(nb_arr)
    arr2 = pyarrow_cuda_buffer_as.numpy_ndarray(pa_cbuf)
    np.testing.assert_array_equal(arr1, arr2)
    assert arr1[1] == 99
