from . import pyarrow_cuda_buffer_as


def random(size):
    import numba.cuda as cuda
    import numpy as np
    arr = np.random.randint(low=0, high=255, size=size,
                            dtype=np.uint8)
    return cuda.to_device(arr)


def numpy_ndarray(nb_arr):
    """Return a copy of numba DeviceNDArray data as a numpy.ndarray.
    """
    return nb_arr.copy_to_host()


def pyarrow_cuda_buffer(nb_arr):
    """Return pyarrow.cuda.CudaBuffer view of a numba DeviceNDArray.
    """
    import pyarrow.cuda as cuda
    ctx = cuda.Context()
    if hasattr(ctx, 'buffer_from_object'):
        # buffer_from_object is defined in arrow>=0.12.1
        return ctx.buffer_from_object(nb_arr)
    desc = nb_arr.__cuda_array_interface__
    addr = desc['data'][0]
    size = nb_arr.alloc_size
    strides = desc.get('strides')
    assert strides in [(1, ), None], repr(strides)
    return ctx.foreign_buffer(addr, size)

def cupy_cuda_MemoryPointer(nb_arr):
    """Return cupy.cuda.MemoryPointer view of a numba DeviceNDArray.
    """
    import cupy
    addr = nb_arr.device_ctypes_pointer.value
    size = nb_arr.alloc_size
    mem = cupy.cuda.UnownedMemory(addr, size, nb_arr)
    return cupy.cuda.MemoryPointer(mem, 0)


def cupy_ndarray(nb_arr):
    """Return cupy.ndarray view of a numba DeviceNDArray.
    """
    import cupy
    return cupy.ndarray(nb_arr.shape, dtype=cupy.uint8,
                        strides=nb_arr.strides,
                        memptr=cupy_cuda_MemoryPointer(nb_arr))


def xnd_xnd_cuda(nb_arr):
    """Return xnd.xnd view of a numba DeviceNDArray.
    """
    cbuf = pyarrow_cuda_buffer(nb_arr)
    # DERIVED
    return pyarrow_cuda_buffer_as.xnd_xnd_cuda(cbuf)


def cudf_Series(nb_arr):
    """Return cudf.Series view of a numba DeviceNDArray.
    """
    import cudf
    return cudf.Series(nb_arr)
