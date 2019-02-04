
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
    """Return pyarrow.cuda.CudaBuffer view of numba DeviceNDArray.
    """
    import pyarrow.cuda as cuda
    ctx = cuda.Context()
    return ctx.buffer_from_object(nb_arr)
