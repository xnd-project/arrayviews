# ArrayViews

## Matrix of supported array views - host memory

The following table summarizes the support of creating a specific array
view (top-row) for the given array storage
objects (left-hand-side column). 

<!--START arrayviews-support_kernel TABLE-->
<table style="width:100%">
<tr><th rowspan=2>Objects</th><th colspan="4">Views</th></tr>
<tr><th>numpy.ndarray</th><th>pandas.Series</th><th>pyarrow.Array</th><th>xnd.xnd</th></tr>
<tr><th>numpy.ndarray</th><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/numpy_ndarray_as.py#L39 title="def pandas_series(arr):
    import pandas as pd
    return pd.Series(arr, copy=False)
">OPTIMAL, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/numpy_ndarray_as.py#L17 title="def pyarrow_array(arr):
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
">GENBITMAP, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/numpy_ndarray_as.py#L46 title="def xnd_xnd(arr):
    import numpy as np
    import xnd
    xd = xnd.xnd.from_buffer(arr)
    if issubclass(arr.dtype.type, (np.floating, np.complexfloating)):
        isnan = np.isnan(arr)
        if isnan.any():
            raise NotImplementedError('xnd view of numpy ndarray with nans')
    return xd
">OPTIMAL, PARTIAL</a></td></tr>
<tr><th>pandas.Series</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pandas_series_as.py#L12 title="def numpy_ndarray(pd_ser):
    return pd_ser.to_numpy()
">OPTIMAL, FULL</a></td><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pandas_series_as.py#L18 title="def pyarrow_array(pd_ser):
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
">GENBITMAP, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pandas_series_as.py#L33 title="def xnd_xnd(pd_ser):
    import xnd
    isnan = pd_ser.isna()
    if not isnan.any():
        return xnd.xnd.from_buffer(pd_ser.to_numpy())
    raise NotImplementedError('xnd view of pandas.Series with nans')
">OPTIMAL, PARTIAL</a></td></tr>
<tr><th>pyarrow.Array</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pyarrow_array_as.py#L11 title="def numpy_ndarray(pa_arr):
    if pa_arr.null_count == 0:
        return pa_arr.to_numpy()
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('numpy.ndarray view of pyarrow.Array with nulls')
">OPTIMAL, PARTIAL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pyarrow_array_as.py#L20 title="def pandas_series(pa_arr):
    import pandas as pd
    if pa_arr.null_count == 0:
        return pd.Series(pa_arr.to_numpy(), copy=False)
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('pandas.Series view of pyarrow.Array with nulls')
">OPTIMAL, PARTIAL</a></td><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pyarrow_array_as.py#L30 title="def xnd_xnd(pa_arr):
    import xnd
    if pa_arr.null_count == 0:
        return xnd.xnd.from_buffer(pa_arr.to_numpy())
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('xnd view of pyarrow.Array with nulls')
">OPTIMAL, PARTIAL</a></td></tr>
<tr><th>xnd.xnd</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/xnd_xnd_as.py#L17 title="def numpy_ndarray(xd_arr):
    import numpy as np
    if not xd_arr.dtype.isoptional():
        return np.array(xd_arr, copy=False)
    raise NotImplementedError(
        'numpy.ndarray view of xnd.xnd with optional values')
">OPTIMAL, PARTIAL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/xnd_xnd_as.py#L27 title="def pandas_series(xd_arr):
    import pandas as pd
    if not xd_arr.dtype.isoptional():
        return pd.Series(memoryview(xd_arr),
                         dtype=str(xd_arr.dtype),
                         copy=False)
    raise NotImplementedError(
        'pandas.Series view of xnd.xnd with optional values')
">OPTIMAL, PARTIAL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/xnd_xnd_as.py#L39 title="def pyarrow_array(xd_arr):
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

### Comments

1. In `numpy.ndarray` and `pandas.Series`, the `numpy.nan` value is interpreted as null value.
2. `OPTIMAL` means that view creation does not require processing of array data
3. `GENBITMAP` means that view creation does requires processing of array data in the presence of null values.
4. `FULL` means that view creation supports the inputs with null values.
5. `PARTIAL` means that view creation does not support the inputs with null values.
6. For the implementation of view constructions, hover over table cell or click on the links to `arrayviews` package source code.

## Efficiency of creating array views

<!--START arrayviews-measure_kernel TABLE-->
<table style="width:100%">
<tr><th rowspan=2>Objects</th><th colspan="4">Views</th></tr>
<tr><th>numpy.ndarray</th><th>pandas.Series</th><th>pyarrow.Array</th><th>xnd.xnd</th></tr>
<tr><th>numpy.ndarray</th><td>0.99(1.0)</td><td>581.3(557.66)</td><td>98.03(712.69)</td><td>58.34(N/A)</td></tr>
<tr><th>pandas.Series</th><td>38.67(38.57)</td><td>1.04(0.98)</td><td>1732.52(2703.66)</td><td>1654.1(N/A)</td></tr>
<tr><th>pyarrow.Array</th><td>16.52(N/A)</td><td>646.92(N/A)</td><td>0.99(1.03)</td><td>39.23(N/A)</td></tr>
<tr><th>xnd.xnd</th><td>17.2(17.15)</td><td>1179.86(1179.52)</td><td>64.71(64.07)</td><td>1.01(1.03)</td></tr>
</table>
<!--END arrayviews-measure_kernel TABLE-->

### Comments

1. The numbers in the table are `<elapsed time to create a view of an obj>/<elapsed time to call 'def dummy(obj): return obj'>`.
2. Results in the parenthesis correspond to objects with nulls.

## Matrix of supported array views - CUDA device memory

TODO
