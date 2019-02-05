def random(size):
    """Return random cupy.ndarray instance of 8 bit insigned integers.
    """
    import cupy
    return cupy.random.randint(0, 256, dtype=cupy.uint8, size=size)


def numpy_ndarray(cp_arr):
    """Return a copy of CudaBuffer data as a numpy.ndarray.
    """
    import cupy
    return cupy.asnumpy(cp_arr)


def numba_cuda_DeviceNDArray(cp_arr):
    """Return numba DeviceNDArray view of cupy.ndarray.
    """
    import numba.cuda as nb_cuda
    return nb_cuda.as_cuda_array(cp_arr)


def pyarrow_cuda_buffer(cp_arr):
    """Return pyarrow.cuda.CudaBuffer view of cupy.ndarray.
    """
    import pyarrow.cuda as cuda
    ctx = cuda.Context(cp_arr.data.device.id)
    return ctx.foreign_buffer(cp_arr.data.ptr, cp_arr.nbytes)


def cupy_cuda_MemoryPointer(cp_arr):
    """Return cupy.cuda.MemoryPointer view of cupy.ndarray.
    """
    return cp_arr.data
