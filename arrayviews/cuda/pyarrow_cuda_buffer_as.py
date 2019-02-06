

def random(size):
    import numpy as np
    import pyarrow as pa
    import pyarrow.cuda as cuda
    ctx = cuda.Context()
    dtype = np.dtype('uint8')
    buf = pa.allocate_buffer(size*dtype.itemsize)
    arr = np.frombuffer(buf, dtype=dtype)
    arr[:] = np.random.randint(low=0, high=255, size=size,
                               dtype=np.uint8)
    cbuf = ctx.new_buffer(buf.size)
    cbuf.copy_from_host(buf, position=0, nbytes=buf.size)
    return cbuf


def numpy_ndarray(cbuf):
    """Return a copy of CudaBuffer data as a numpy.ndarray.
    """
    import numpy as np
    dtype = np.dtype('uint8')
    return np.frombuffer(cbuf.copy_to_host(), dtype=dtype)


def numba_cuda_DeviceNDArray(cbuf):
    """Return numba DeviceNDArray view of a pyarrow.cuda.CudaBuffer.
    """
    import numpy as np
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    dtype = np.dtype('uint8')
    return DeviceNDArray((cbuf.size,), (dtype.itemsize,), dtype,
                         gpu_data=cbuf.to_numba())


def cupy_cuda_MemoryPointer(cbuf):
    """Return cupy.cuda.MemoryPointer view of a pyarrow.cuda.CudaBuffer.
    """
    import cupy
    #addr = cbuf.context.get_device_address(cbuf.address) # requires arrow>=0.12.1
    addr = cbuf.address
    mem = cupy.cuda.UnownedMemory(addr, cbuf.size, cbuf)
    return cupy.cuda.MemoryPointer(mem, 0)


def cupy_ndarray(cbuf):
    """Return cupy.ndarray view of a pyarrow.cuda.CudaBuffer.
    """
    import cupy
    return cupy.ndarray(cbuf.size, dtype=cupy.uint8,
                        memptr=cupy_cuda_MemoryPointer(cbuf))


def xnd_xnd_cuda(cbuf):
    """Return xnd.xnd view of a pyarrow.cuda.CudaBuffer [EXPERIMENTAL].
    """
    import xnd
    import pyarrow as pa
    #addr = cbuf.context.get_device_address(cbuf.address) # requires arrow>=0.12.1
    addr = cbuf.address
    # device = cbuf.context.device_number
    buf = pa.foreign_buffer(addr, cbuf.size, cbuf)
    return xnd.xnd.from_buffer(buf)
