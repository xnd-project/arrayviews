def numpy_ndarray(xd_arr):
    """Return numpy.ndarray view of a xnd.xnd
    """
    import xnd
    import numpy as np
    if not xd_arr.dtype.isoptional():
        return np.frombuffer(memoryview(xd_arr), dtype=str(xd_arr.dtype))
    raise NotImplementedError('numpy.ndarray view of xnd with optional values')

def pandas_series(xd_arr):
    """Return pandas.Series view of a xnd.xnd
    """
    import xnd
    import pandas as pd
    if not xd_arr.dtype.isoptional():
        return pd.Series(memoryview(xd_arr),
                         dtype=str(xd_arr.dtype),
                         copy=False)
    raise NotImplementedError('pandas.Series view of xnd with optional values')

def pyarrow_array(xd_arr):
    """Return pyarrow.Array view of a xnd.xnd
    """
    import xnd
    import pyarrow as pa
    if not xd_arr.dtype.isoptional():
        pa_buf = pa.py_buffer(memoryview(xd_arr))
        return pa.Array.from_buffers(
            pa.from_numpy_dtype(str(xd_arr.dtype)),
            xd_arr.type.datasize//xd_arr.type.itemsize,
            [None, pa_buf])
    raise NotImplementedError('pyarrow.Array view of xnd with optional values')
