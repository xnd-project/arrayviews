from .utils import get_bitmap
from . import numpy_ndarray_as


def random(size, nulls=False):
    """Return random pandas.Series instance of 64 bit floats.
    """
    return numpy_ndarray_as.pandas_series(
        numpy_ndarray_as.random(size, nulls=nulls),
        nan_to_null=True)


def numpy_ndarray(pd_ser, nan_to_null=False):
    """Return numpy.ndarray view of a pandas.Series
    """
    return pd_ser.to_numpy()


def pyarrow_array(pd_ser, nan_to_null=False):
    """Return pyarrow.Array view of a pandas.Series
    """
    import numpy as np
    import pyarrow as pa
    if nan_to_null and issubclass(pd_ser.dtype.type,
                                  (np.floating, np.complexfloating)):
        isnan = pd_ser.isna()
        if isnan.any():
            pa_nul = pa.py_buffer(get_bitmap(isnan.to_numpy()))
            return pa.Array.from_buffers(pa.from_numpy_dtype(pd_ser.dtype),
                                         pd_ser.size,
                                         [pa_nul,
                                          pa.py_buffer(pd_ser.to_numpy())])
    return pa.Array.from_buffers(pa.from_numpy_dtype(pd_ser.dtype),
                                 pd_ser.size,
                                 [None, pa.py_buffer(pd_ser.to_numpy())])


def xnd_xnd(pd_ser, nan_to_null=False):
    """Return xnd view of a pandas.Series
    """
    import numpy as np
    import xnd
    if nan_to_null and issubclass(pd_ser.dtype.type,
                                  (np.floating, np.complexfloating)):
        isnan = pd_ser.isna()
        if isnan.any():
            raise NotImplementedError('xnd view of pandas.Series with nans')
    return xnd.xnd.from_buffer(pd_ser.to_numpy())
