from . import numpy_ndarray_as


def random(size, nulls=False):
    """Return random pyarrow.Array instance of 64 bit floats.
    """
    return numpy_ndarray_as.pyarrow_array(
        numpy_ndarray_as.random(size, nulls=nulls))


def numpy_ndarray(pa_arr):
    """Return numpy.ndarray view of a pyarrow.Array
    """
    if pa_arr.null_count == 0:
        return pa_arr.to_numpy()
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('numpy.ndarray view of pyarrow.Array with nulls')


def pandas_series(pa_arr):
    """Return pandas.Series view of a pyarrow.Array
    """
    import pandas as pd
    if pa_arr.null_count == 0:
        return pd.Series(pa_arr.to_numpy(), copy=False)
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('pandas.Series view of pyarrow.Array with nulls')


def xnd_xnd(pa_arr):
    """Return xnd view of a pyarrow.Array
    """
    import xnd
    if pa_arr.null_count == 0:
        return xnd.xnd.from_buffer(pa_arr.to_numpy())
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('xnd view of pyarrow.Array with nulls')
