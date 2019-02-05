from . import cupy_ndarray_as


def random(size):
    """Return random cupy.cuda.MemoryPointer instance of 8 bit insigned integers.
    """
    cp_ptr = cupy_ndarray_as.random(size).data
    assert cp_ptr.mem.size == size, repr((cp_ptr.mem.size, size))
    return cp_ptr


def numpy_ndarray(cp_ptr):
    """Return a copy of cupy.cuda.MemoryPointer data as a numpy.ndarray.
    """
    import numpy as np
    buf = np.empty((cp_ptr.mem.size), np.uint8)
    descr = buf.__array_interface__()
    addr = descr['data'][0]
    cp_ptr.copy_to_host(addr, buf.size)
    return buf
