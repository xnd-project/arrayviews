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
    import numpy as np
    import pandas as pd
    if not xd_arr.dtype.isoptional():
        return pd.Series(np.array(xd_arr, copy=False), copy=False)
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

#### Comments

1. In `numpy.ndarray` and `pandas.Series`, the `numpy.nan` value is interpreted as null value.
2. `OPTIMAL` means that view creation does not require processing of array data
3. `GENBITMAP` means that view creation does requires processing of array data in the presence of null or nan values. By default, such processing is disabled.
4. `FULL` means that view creation supports the inputs with null values.
5. `PARTIAL` means that view creation does not support the inputs with null values.
6. For the implementation of view constructions, hover over table cell or click on the links to `arrayviews` package source code.

### Benchmark: creating array views - host memory

<!--START arrayviews-measure_kernel TABLE-->
<table style="width:100%">
<tr><th rowspan=2>Objects</th><th colspan="4">Views</th></tr>
<tr><th>numpy.ndarray</th><th>pandas.Series</th><th>pyarrow.Array</th><th>xnd.xnd</th></tr>
<tr><th>numpy.ndarray</th><td>1.03(0.99)</td><td>326.28(326.9)</td><td>31.52(31.7)</td><td>15.79(15.61)</td></tr>
<tr><th>pandas.Series</th><td>30.07(30.07)</td><td>1.01(1.01)</td><td>79.67(79.83)</td><td>48.25(47.86)</td></tr>
<tr><th>pyarrow.Array</th><td>17.28(N/A)</td><td>348.53(N/A)</td><td>0.98(0.96)</td><td>35.35(N/A)</td></tr>
<tr><th>xnd.xnd</th><td>13.85(N/A)</td><td>340.09(N/A)</td><td>50.96(N/A)</td><td>0.98(0.97)</td></tr>
</table>
<!--END arrayviews-measure_kernel TABLE-->

#### Comments

1. The numbers in the table are `<elapsed time to create a view of an obj>/<elapsed time to call 'def dummy(obj): return obj'>`.
2. Results in the parenthesis correspond to objects with nulls or nans. No attempts are made to convert nans to nulls. 
3. Test arrays are 64-bit float arrays of size 10000.

## Matrix of supported array views - CUDA device memory

<!--START arrayviews.cuda-support_kernel TABLE-->
<table style="width:100%">
<tr><th rowspan=2>Objects</th><th colspan="2">Views</th></tr>
<tr><th>numba DeviceNDArray</th><th>pyarrow.cuda.CudaBuffer</th></tr>
<tr><th>numba DeviceNDArray</th><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/numba_cuda_DeviceNDArray_as.py#L16 title="def pyarrow_cuda_buffer(nb_arr):
    import pyarrow.cuda as cuda
    ctx = cuda.Context()
    return ctx.buffer_from_object(nb_arr)
">OPTIMAL, FULL</a></td></tr>
<tr><th>pyarrow.cuda.CudaBuffer</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/pyarrow_cuda_buffer_as.py#L25 title="def numba_cuda_DeviceNDArray(cbuf):
    import numpy as np
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    dtype = np.dtype('uint8')
    return DeviceNDArray((cbuf.size,), (dtype.itemsize,), dtype,
                         gpu_data=cbuf.to_numba())
">OPTIMAL, FULL</a></td><td></td></tr>
</table>
<!--END arrayviews.cuda-support_kernel TABLE-->

### Benchmark: creating array views - CUDA device memory

<!--START arrayviews.cuda-measure_kernel TABLE-->
<table style="width:100%">
<tr><th rowspan=2>Objects</th><th colspan="2">Views</th></tr>
<tr><th>numba DeviceNDArray</th><th>pyarrow.cuda.CudaBuffer</th></tr>
<tr><th>numba DeviceNDArray</th><td>1.01</td><td>84.51</td></tr>
<tr><th>pyarrow.cuda.CudaBuffer</th><td>370.91</td><td>0.99</td></tr>
</table>
<!--END arrayviews.cuda-measure_kernel TABLE-->

#### Comments

1. Test arrays are 8-bit unsigned integer arrays of size 10000.
