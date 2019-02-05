import pytest
from arrayviews.cuda import xnd_xnd_cuda_as
from arrayviews.cuda import pyarrow_cuda_buffer_as

xnd = pytest.importorskip("xnd")
np = pytest.importorskip("numpy")

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
    xd_arr = xnd_xnd_cuda_as.random(5)
    pa_cbuf = xnd_xnd_cuda_as.pyarrow_cuda_buffer(xd_arr)
    arr1 = xnd_xnd_cuda_as.numpy_ndarray(xd_arr)
    arr2 = pyarrow_cuda_buffer_as.numpy_ndarray(pa_cbuf)
    np.testing.assert_array_equal(arr1, arr2)

    pa_cbuf.copy_from_host(np.array([99]), 1, 1)
    arr1 = xnd_xnd_cuda_as.numpy_ndarray(xd_arr)
    arr2 = pyarrow_cuda_buffer_as.numpy_ndarray(pa_cbuf)
    np.testing.assert_array_equal(arr1, arr2)
    assert arr1[1] == 99
