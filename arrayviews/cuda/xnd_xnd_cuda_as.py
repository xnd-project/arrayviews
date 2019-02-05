from . import pyarrow_cuda_buffer_as


def random(size):
    """Return random xnd.xnd instance of 8 bit ints on CUDA device.
    """
    cbuf = pyarrow_cuda_buffer_as.random(size)
    return pyarrow_cuda_buffer_as.xnd_xnd_cuda(cbuf)


def numpy_ndarray(xd_arr):
    """Return numpy.ndarray view of a xnd.xnd on CUDA device.
    """
    cbuf = pyarrow_cuda_buffer(xd_arr)
    return pyarrow_cuda_buffer_as.numpy_ndarray(cbuf)


def pyarrow_cuda_buffer(xd_arr):
    """Return pyarrow.cuda.CudaBuffer view of a xnd.xnd in CUDA device.
    """
    import pyarrow as pa
    import pyarrow.cuda as cuda
    buf = pa.py_buffer(memoryview(xd_arr))
    ctx = cuda.Context()
    return ctx.foreign_buffer(buf.address, buf.size)


def cupy_cuda_MemoryPointer(xd_arr):
    """Return cupy.cuda.MemoryPointer view of a xnd.xnd in CUDA device.
    """
    import cupy
    import pyarrow as pa
    buf = pa.py_buffer(memoryview(xd_arr))
    mem = cupy.cuda.UnownedMemory(buf.address, buf.size, xd_arr)
    return cupy.cuda.MemoryPointer(mem, 0)


def cupy_ndarray(xd_arr):
    """Return cupy.ndarray view of a xnd.xnd in CUDA device.
    """
    import cupy
    return cupy.ndarray(xd_arr.type.datasize, dtype=cupy.uint8,
                        memptr=cupy_cuda_MemoryPointer(xd_arr))


def numba_cuda_DeviceNDArray(xd_arr):
    """Return cupy.ndarray view of a xnd.xnd in CUDA device.
    """
    cbuf = pyarrow_cuda_buffer(xd_arr)
    # DERIVED
    return pyarrow_cuda_buffer_as.numba_cuda_DeviceNDArray(cbuf)
