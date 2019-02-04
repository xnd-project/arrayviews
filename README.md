# ArrayViews

## Matrix of supported array views - host memory

The following table summarizes the support of creating a specific array
view (left-hand-side column) of given array storage
object (top-row). The table cells contains two answers
corresponding to if the view can be created from an array object without or with nulls, respectively.
In the case of `numpy.ndarray`, the `nan` values are interpreted as nulls.

<!--START arrayviews TABLE-->
<table style="width:100%">
<tr><th>Views</th><th colspan="4"></th></tr>
<tr><th></th><th>numpy_ndarray</th><th>pandas_series</th><th>pyarrow_array</th><th>xnd_xnd</th></tr>
<tr><th>numpy_ndarray</th><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/numpy_ndarray_to.py#L27 title="def pandas_series(arr):
    import pandas as pd
    return pd.Series(arr, copy=False)
">SUPPORTED</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/numpy_ndarray_to.py#L5 title="def pyarrow_array(arr):
    import numpy as np
    import pyarrow as pa
    if issubclass(arr.dtype.type, (np.floating, np.complexfloating)):
        isnan = np.isnan(arr)
        if isnan.any():
            pa_nul = pa.py_buffer(get_bitmap(isnan))
            return pa.Array.from_buffers(pa.from_numpy_dtype(arr.dtype),
                                         arr.size,
                                         [pa_nul, pa.py_buffer(arr)])
    return pa.Array.from_buffers(pa.from_numpy_dtype(arr.dtype),
                                 arr.size,
                                 [None, pa.py_buffer(arr)])
">SUPPORTED</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/numpy_ndarray_to.py#L34 title="def xnd_xnd(arr):
    import numpy as np
    import xnd
    xd = xnd.xnd.from_buffer(arr)
    if issubclass(arr.dtype.type, (np.floating, np.complexfloating)):
        isnan = np.isnan(arr)
        if isnan.any():
            raise NotImplementedError('xnd view of numpy ndarray with nans')
    return xd
">SUPPORTED</a></td></tr>
<tr><th>pandas_series</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pandas_series_to.py#L3 title="def numpy_ndarray(pd_ser):
    import pandas as pd
    import numpy as np
    return pd_ser.to_numpy()
">SUPPORTED</a></td><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pandas_series_to.py#L10 title="def pyarrow_array(pd_ser):
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
">SUPPORTED</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pandas_series_to.py#L25 title="def xnd_xnd(pd_ser):
    import pandas as pd
    import xnd
    isnan = pd_ser.isna()
    if not isnan.any():
        return xnd.xnd.from_buffer(pd_ser.to_numpy())
    raise NotImplementedError('xnd view of pandas.Series with nans')
">SUPPORTED</a></td></tr>
<tr><th>pyarrow_array</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pyarrow_array_to.py#L2 title="def numpy_ndarray(pa_arr):
    import pyarrow as pa
    import numpy as np
    return pa_arr.to_numpy()
">SUPPORTED</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pyarrow_array_to.py#L9 title="def pandas_series(pa_arr):
    import pyarrow as pa
    import pandas as pd
    return pd.Series(pa_arr.to_pandas(), copy=False)
">SUPPORTED</a></td><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pyarrow_array_to.py#L16 title="def xnd_xnd(pa_arr):
    import pyarrow as pa
    import xnd
    if pa_arr.null_count == 0:
        return xnd.xnd.from_buffer(pa_arr.to_numpy())
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('xnd view of pyarrow.Array with nulls')
">SUPPORTED</a></td></tr>
<tr><th>xnd_xnd</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/xnd_xnd_to.py#L1 title="def numpy_ndarray(xd_arr):
    import xnd
    import numpy as np
    if not xd_arr.dtype.isoptional():
        return np.frombuffer(memoryview(xd_arr), dtype=str(xd_arr.dtype))
    raise NotImplementedError('numpy.ndarray view of xnd with optional values')
">SUPPORTED</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/xnd_xnd_to.py#L10 title="def pandas_series(xd_arr):
    import xnd
    import pandas as pd
    if not xd_arr.dtype.isoptional():
        return pd.Series(memoryview(xd_arr),
                         dtype=str(xd_arr.dtype),
                         copy=False)
    raise NotImplementedError('pandas.Series view of xnd with optional values')
">SUPPORTED</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/xnd_xnd_to.py#L21 title="def pyarrow_array(xd_arr):
    import xnd
    import pyarrow as pa
    if not xd_arr.dtype.isoptional():
        pa_buf = pa.py_buffer(memoryview(xd_arr))
        return pa.Array.from_buffers(
            pa.from_numpy_dtype(str(xd_arr.dtype)),
            xd_arr.type.datasize//xd_arr.type.itemsize,
            [None, pa_buf])
    raise NotImplementedError('pyarrow.Array view of xnd with optional values')
">SUPPORTED</a></td><td></td></tr>
</table>
<!--END arrayviews TABLE-->


For the implementation of view constructions, see `arrayviews` package source code.

## Matrix of supported array views - CUDA device memory

TODO
