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
    return ctx.buffer_from_object(nb_arr)


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
