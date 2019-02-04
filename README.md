# ArrayViews

## Matrix of supported array views - host memory

The following table summarizes the support of creating a specific array
view (top-row) for the given array storage
objects (left-hand-side column). 

<!--START arrayviews-support_kernel TABLE-->
<table style="width:100%">
<tr><th rowspan=2>Objects</th><th colspan="4">Views</th></tr>
<tr><th>numpy.ndarray</th><th>pandas.Series</th><th>pyarrow.Array</th><th>xnd.xnd</th></tr>
<tr><th>numpy.ndarray</th><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/numpy_ndarray_as.py#L40 title="def pandas_series(arr, nan_to_null=False):
    import pandas as pd
    return pd.Series(arr, copy=False)
">OPTIMAL, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/numpy_ndarray_as.py#L17 title="def pyarrow_array(arr, nan_to_null=False):
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
">GENBITMAP, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/numpy_ndarray_as.py#L47 title="def xnd_xnd(arr, nan_to_null=False):
    import numpy as np
    import xnd
    xd = xnd.xnd.from_buffer(arr)
    if nan_to_null and issubclass(arr.dtype.type,
                                  (np.floating, np.complexfloating)):
        isnan = np.isnan(arr)
        if isnan.any():
            raise NotImplementedError('xnd view of numpy ndarray with nans')
    return xd
">OPTIMAL, PARTIAL</a></td></tr>
<tr><th>pandas.Series</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pandas_series_as.py#L13 title="def numpy_ndarray(pd_ser, nan_to_null=False):
    return pd_ser.to_numpy()
">OPTIMAL, FULL</a></td><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pandas_series_as.py#L19 title="def pyarrow_array(pd_ser, nan_to_null=False):
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
">GENBITMAP, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pandas_series_as.py#L38 title="def xnd_xnd(pd_ser, nan_to_null=False):
    import numpy as np
    import xnd
    if nan_to_null and issubclass(pd_ser.dtype.type,
                                  (np.floating, np.complexfloating)):
        isnan = pd_ser.isna()
        if isnan.any():
            raise NotImplementedError('xnd view of pandas.Series with nans')
    return xnd.xnd.from_buffer(pd_ser.to_numpy())
">OPTIMAL, PARTIAL</a></td></tr>
<tr><th>pyarrow.Array</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pyarrow_array_as.py#L12 title="def numpy_ndarray(pa_arr):
    if pa_arr.null_count == 0:
        return pa_arr.to_numpy()
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('numpy.ndarray view of pyarrow.Array with nulls')
">OPTIMAL, PARTIAL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pyarrow_array_as.py#L21 title="def pandas_series(pa_arr):
    import pandas as pd
    if pa_arr.null_count == 0:
        return pd.Series(pa_arr.to_numpy(), copy=False)
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('pandas.Series view of pyarrow.Array with nulls')
">OPTIMAL, PARTIAL</a></td><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pyarrow_array_as.py#L31 title="def xnd_xnd(pa_arr):
    import xnd
    if pa_arr.null_count == 0:
        return xnd.xnd.from_buffer(pa_arr.to_numpy())
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('xnd view of pyarrow.Array with nulls')
">OPTIMAL, PARTIAL</a></td></tr>
<tr><th>xnd.xnd</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/xnd_xnd_as.py#L18 title="def numpy_ndarray(xd_arr):
    import numpy as np
    if not xd_arr.dtype.isoptional():
        return np.array(xd_arr, copy=False)
    raise NotImplementedError(
        'numpy.ndarray view of xnd.xnd with optional values')
">OPTIMAL, PARTIAL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/xnd_xnd_as.py#L28 title="def pandas_series(xd_arr):
    import pandas as pd
    if not xd_arr.dtype.isoptional():
        return pd.Series(memoryview(xd_arr),
                         dtype=str(xd_arr.dtype),
                         copy=False)
    raise NotImplementedError(
        'pandas.Series view of xnd.xnd with optional values')
">OPTIMAL, PARTIAL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/xnd_xnd_as.py#L40 title="def pyarrow_array(xd_arr):
    import pyarrow as pa
    if not xd_arr.dtype.isoptional():
        pa_buf = pa.py_buffer(memoryview(xd_arr))
        return pa.Array.from_buffers(
            pa.from_numpy_dtype(str(xd_arr.dtype)),
            xd_arr.type.datasize//xd_arr.type.itemsize,
            [None, pa_buf])
    raise NotImplementedError(
        'pyarrow.Array view of xnd.xnd with optional values')
">OPTIMAL, PARTIAL</a></td><td></td></tr>
</table>
<!--END arrayviews-support_kernel TABLE-->

#### Comments

1. In `numpy.ndarray` and `pandas.Series`, the `numpy.nan` value is interpreted as null value.
2. `OPTIMAL` means that view creation does not require processing of array data
3. `GENBITMAP` means that view creation does requires processing of array data in the presence of null or nan values. By default, such processing is disabled.
4. `FULL` means that view creation supports the inputs with null values.
5. `PARTIAL` means that view creation does not support the inputs with null values.
6. For the implementation of view constructions, hover over table cell or click on the links to `arrayviews` package source code.

### Benchmark: creating array views

<!--START arrayviews-measure_kernel TABLE-->
<table style="width:100%">
<tr><th rowspan=2>Objects</th><th colspan="4">Views</th></tr>
<tr><th>numpy.ndarray</th><th>pandas.Series</th><th>pyarrow.Array</th><th>xnd.xnd</th></tr>
<tr><th>numpy.ndarray</th><td>0.99(1.0)</td><td>577.15(551.05)</td><td>35.95(32.05)</td><td>15.97(15.52)</td></tr>
<tr><th>pandas.Series</th><td>38.63(38.51)</td><td>1.01(0.92)</td><td>100.86(98.74)</td><td>57.67(57.56)</td></tr>
<tr><th>pyarrow.Array</th><td>17.02(N/A)</td><td>648.22(N/A)</td><td>0.99(0.98)</td><td>36.59(N/A)</td></tr>
<tr><th>xnd.xnd</th><td>15.85(N/A)</td><td>1162.02(N/A)</td><td>58.78(N/A)</td><td>0.97(0.99)</td></tr>
</table>
<!--END arrayviews-measure_kernel TABLE-->

#### Comments

1. The numbers in the table are `<elapsed time to create a view of an obj>/<elapsed time to call 'def dummy(obj): return obj'>`.
2. Results in the parenthesis correspond to objects with nulls or nans. No attempts are made to convert nans to nulls. 
3. Test arrays are 64-bit float arrays of size 10000.

## Matrix of supported array views - CUDA device memory

TODO
