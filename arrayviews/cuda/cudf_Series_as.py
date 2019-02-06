from . import numba_cuda_DeviceNDArray_as


def random(size):
    import cudf
    return cudf.Series(numba_cuda_DeviceNDArray_as.random(size))


def numpy_ndarray(cf_ser):
    return cf_ser.to_array()


def numba_cuda_DeviceNDArray(cf_ser):
    return cf_ser.data.mem
