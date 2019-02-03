from .utils import get_bitmap

def numpy_ndarray(pd_ser):
    """Return numpy.ndarray view of a pandas.Series
    """
    import pandas as pd
    import numpy as np
    return pd_ser.to_numpy()

def pyarrow_array(pd_ser):
    """Return pyarrow.Array view of a pandas.Series
    """
    import pandas as pd
    import pyarrow as pa
    isnan = pd_ser.isna()
    if isnan.any():
        pa_nul = pa.py_buffer(get_bitmap(isnan.to_numpy()))
        return pa.Array.from_buffers(pa.from_numpy_dtype(pd_ser.dtype),
                                     pd_ser.size,
                                     [pa_nul, pa.py_buffer(pd_ser.to_numpy())])
    return pa.Array.from_buffers(pa.from_numpy_dtype(pd_ser.dtype),
                                 pd_ser.size,
                                 [None, pa.py_buffer(pd_ser.to_numpy())])

def xnd_xnd(pd_ser):
    """Return xnd view of a pandas.Series
    """
    import pandas as pd
    import xnd
    isnan = pd_ser.isna()
    if not isnan.any():
        return xnd.xnd.from_buffer(pd_ser.to_numpy())
    raise NotImplementedError('xnd view of pandas.Series with nans')
