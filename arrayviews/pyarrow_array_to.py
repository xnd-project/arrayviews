
def numpy_ndarray(pa_arr):
    """Return numpy.ndarray view of a pyarrow.Array
    """
    import pyarrow as pa
    import numpy as np
    return pa_arr.to_numpy()

def pandas_series(pa_arr):
    """Return pandas.Series view of a pyarrow.Array
    """
    import pyarrow as pa
    import pandas as pd
    return pd.Series(pa_arr.to_pandas(), copy=False)
    
def xnd_xnd(pa_arr):
    """Return xnd view of a pyarrow.Array
    """
    import pyarrow as pa
    import xnd
    if pa_arr.null_count == 0:
        return xnd.xnd.from_buffer(pa_arr.to_numpy())
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('xnd view of pyarrow.Array with nulls')
