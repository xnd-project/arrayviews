"""Viewers of numpy ndarray
"""
from .utils import get_bitmap


def random(size, nulls=False):
    """Return random numpy.ndarray instance of 64 bit floats.
    """
    import numpy as np
    r = np.random.random(size)
    if nulls:
        idx = np.random.randint(0, r.size, size=max(1, r.size//4))
        r[idx] = np.nan
    return r


def pyarrow_array(arr, nan_to_null=False):
    """Return pyarrow.Array view of a numpy ndarray.

    In floating arrays, all nan values are interpreted as nulls.

    In complex arrays, if real or imaginary part of an array item
    value is nan, the value is interpreted as null.
    """
    import numpy as np
    import pyarrow as pa
    if nan_to_null and issubclass(arr.dtype.type,
                                  (np.floating, np.complexfloating)):
        isnan = np.isnan(arr)
        if isnan.any():
            pa_nul = pa.py_buffer(get_bitmap(isnan))
            return pa.Array.from_buffers(pa.from_numpy_dtype(arr.dtype),
                                         arr.size,
                                         [pa_nul, pa.py_buffer(arr)])
    return pa.Array.from_buffers(pa.from_numpy_dtype(arr.dtype),
                                 arr.size,
                                 [None, pa.py_buffer(arr)])


def pandas_series(arr, nan_to_null=False):
    """Return pandas.Series view of a numpy ndarray.
    """
    import pandas as pd
    return pd.Series(arr, copy=False)


def xnd_xnd(arr, nan_to_null=False):
    """Return xnd.xnd view of a numpy ndarray.
    """
    import numpy as np
    import xnd
    xd = xnd.xnd.from_buffer(arr)
    if nan_to_null and issubclass(arr.dtype.type,
                                  (np.floating, np.complexfloating)):
        isnan = np.isnan(arr)
        if isnan.any():
            raise NotImplementedError('xnd view of numpy ndarray with nans')
    return xd
