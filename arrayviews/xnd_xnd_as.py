from . import numpy_ndarray_as


def random(size, nulls=False):
    """Return random xnd.xnd instance of 64 bit floats.
    """
    import xnd
    import numpy as np
    r = numpy_ndarray_as.random(size, nulls=nulls)
    if nulls:
        xr = xnd.xnd(r.tolist(), dtype='?float64')
        for i in np.where(np.isnan(r))[0]:
            xr[i] = None
        return xr
    return xnd.xnd(r.tolist(), dtype='float64')


def numpy_ndarray(xd_arr):
    """Return numpy.ndarray view of a xnd.xnd
    """
    import numpy as np
    if not xd_arr.dtype.isoptional():
        return np.array(xd_arr, copy=False)
    raise NotImplementedError(
        'numpy.ndarray view of xnd.xnd with optional values')


def pandas_series(xd_arr):
    """Return pandas.Series view of a xnd.xnd
    """
    import numpy as np
    import pandas as pd
    if not xd_arr.dtype.isoptional():
        return pd.Series(np.array(xd_arr, copy=False), copy=False)
    raise NotImplementedError(
        'pandas.Series view of xnd.xnd with optional values')


def pyarrow_array(xd_arr):
    """Return pyarrow.Array view of a xnd.xnd
    """
    import pyarrow as pa
    if not xd_arr.dtype.isoptional():
        pa_buf = pa.py_buffer(memoryview(xd_arr))
        return pa.Array.from_buffers(
            pa.from_numpy_dtype(str(xd_arr.dtype)),
            xd_arr.type.datasize//xd_arr.type.itemsize,
            [None, pa_buf])
    raise NotImplementedError(
        'pyarrow.Array view of xnd.xnd with optional values')
