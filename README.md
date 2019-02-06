# ArrayViews

## Introduction

There exists many array libraries that implement objects for storing
data in allocated memory areas. Already in Python ecosystem, the
number of such libraries is more than just few (see below), some of
them are designed for referencing the memory of both host RAM and the
memory of accelerator devices such as GPUs. Such Python packages
implement various computational algorithms that one would wish to
apply on the data stored in some other array object than what the
algoritms use. 

Many of the array object implementations support Python Buffer
Protocol [PEP-3118](https://www.python.org/dev/peps/pep-3118/) that
makes it possible to create array objects from other implementations
of array objects without actually copying the memory - this is called
*creating array views*.

As a side note, unfortunately, the Python Buffer Protocol is
incomplete when considering data storage in devices memory. The
PEP-3118 lacks the device concept which makes it almost impossible to
use existing array storage implementations to hold the memory pointers
of such devices.  This has resulted in a emergence of a number of new
array libraries specifically designed for holding pointers to device
memory. However, the approach of reimplementing the array storage
objects for each different device from scatch does not scale well as
the only essential restriction is about the interpretation of a memory
pointer - whether the pointer value can be dereferenced in a (host or
device) process to use the data, or not. The rest of the array object
implementation would remain the same.  Instead, the Python Buffer
Protocol should be extended with the device concept. Hopefully we'll
see it happen in future. Meanwhile...

The aim of this project is to establish a connection between different
data storage object implementations while avoiding copying the data in
host or device memory. The following packages are supported:

| Package | Tested versions | Storage on host | Storage on CUDA device |
|---------|-----------------|-----------------|------------------------|
| numpy   | 1.16.1          | ndarray         | N/A                    |
| pandas  | 0.24.1          | Series          | N/A                    |
| pyarrow | 0.12.1.dev120+g7f9... | Array     | cuda.CudaBuffer        |
| xnd     | 0.2.0dev3       | xnd             | xnd                    |
| cupy    | 5.2.0           | N/A             | ndarray, cuda.MemoryPointer |

## Basic usage

To use ``arrayviews`` package for host memory, import the needed data
storage support modules, for instance,
```
from arrayviews import (
  numpy_ndarray_as,
  pandas_series_as,
  pyarrow_array_as,
  xnd_xnd_as
  )
```
For CUDA based device memory, one can use the following import statement:
```
from arrayviews.cuda import (
  cupy_ndarray_as,
  numba_cuda_DeviceNDArray,
  pyarrow_cuda_buffer_as,
  xnd_xnd_cuda_as
  )
...
```
The general pattern of creating a specific view of another storage object is:
```
data_view = <data storage object>_as.<view data storage object>(data)
```
For example,
```
>>> import numpy as np
>>> import pyarrow as pa
>>> from arrayviews import numpy_ndarray_as
>>> np_arr = np.arange(5)
>>> pa_arr = numpy_ndarray_as.pyarrow_array(np_arr)
>>> print(pa_arr)
[
  0,
  1,
  2,
  3,
  4
]
>>> np_arr[2] = 999    # change numpy array
>>> print(pa_arr)
[
  0,
  1,
  999,
  3,
  4
]
```

## Supported array views - host memory

The following table summarizes the support of creating a specific
array view (top-row) for the given array storage objects
(left-hand-side column).

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
        # TODO: would memoryview.cast approach be more efficient? see xnd_xnd.
        return pa_arr.to_numpy()
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('numpy.ndarray view of pyarrow.Array with nulls')
">OPTIMAL, PARTIAL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pyarrow_array_as.py#L22 title="def pandas_series(pa_arr):
    import pandas as pd
    if pa_arr.null_count == 0:
        return pd.Series(numpy_ndarray(pa_arr), copy=False)
    pa_nul, pa_buf = pa_arr.buffers()
    raise NotImplementedError('pandas.Series view of pyarrow.Array with nulls')
">OPTIMAL, PARTIAL</a></td><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/pyarrow_array_as.py#L32 title="def xnd_xnd(pa_arr):
    import xnd
    if pa_arr.null_count == 0:
        import numpy as np
        pa_nul, pa_buf = pa_arr.buffers()
        dtype = np.dtype(pa_arr.type.to_pandas_dtype())
        return xnd.xnd.from_buffer(memoryview(pa_buf).cast(dtype.char,
                                                           (len(pa_arr),)))
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
<tr><th>numpy.ndarray</th><td>1.02(0.98)</td><td>322.66(323.84)</td><td>31.92(31.54)</td><td>15.6(15.4)</td></tr>
<tr><th>pandas.Series</th><td>30.21(30.21)</td><td>0.98(0.98)</td><td>79.62(78.66)</td><td>47.42(47.95)</td></tr>
<tr><th>pyarrow.Array</th><td>16.97(N/A)</td><td>356.83(N/A)</td><td>0.99(0.99)</td><td>26.26(N/A)</td></tr>
<tr><th>xnd.xnd</th><td>14.22(N/A)</td><td>345.06(N/A)</td><td>52.4(N/A)</td><td>0.96(0.95)</td></tr>
</table>
<!--END arrayviews-measure_kernel TABLE-->

#### Comments

1. The numbers in the table are `<elapsed time to create a view of an obj>/<elapsed time to call 'def dummy(obj): return obj'>`.
2. Results in the parenthesis correspond to objects with nulls or nans. No attempts are made to convert nans to nulls. 
3. Test arrays are 64-bit float arrays of size 51200.

## Supported array views - CUDA device memory

<!--START arrayviews.cuda-support_kernel TABLE-->
<table style="width:100%">
<tr><th rowspan=2>Objects</th><th colspan="5">Views</th></tr>
<tr><th>pyarrow CudaBuffer</th><th>numba DeviceNDArray</th><th>cupy.ndarray</th><th>cupy MemoryPointer</th><th>xnd.xnd CUDA</th></tr>
<tr><th>pyarrow CudaBuffer</th><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/pyarrow_cuda_buffer_as.py#L26 title="def numba_cuda_DeviceNDArray(cbuf):
    import numpy as np
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    dtype = np.dtype('uint8')
    return DeviceNDArray((cbuf.size,), (dtype.itemsize,), dtype,
                         gpu_data=cbuf.to_numba())
">OPTIMAL, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/pyarrow_cuda_buffer_as.py#L45 title="def cupy_ndarray(cbuf):
    import cupy
    return cupy.ndarray(cbuf.size, dtype=cupy.uint8,
                        memptr=cupy_cuda_MemoryPointer(cbuf))
">OPTIMAL, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/pyarrow_cuda_buffer_as.py#L36 title="def cupy_cuda_MemoryPointer(cbuf):
    import cupy
    addr = cbuf.context.get_device_address(cbuf.address)
    mem = cupy.cuda.UnownedMemory(addr, cbuf.size, cbuf)
    return cupy.cuda.MemoryPointer(mem, 0)
">OPTIMAL, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/pyarrow_cuda_buffer_as.py#L53 title="def xnd_xnd_cuda(cbuf):
    import xnd
    import pyarrow as pa
    addr = cbuf.context.get_device_address(cbuf.address)
    # device = cbuf.context.device_number
    buf = pa.foreign_buffer(addr, cbuf.size, cbuf)
    return xnd.xnd.from_buffer(buf)
">OPTIMAL, FULL</a></td></tr>
<tr><th>numba DeviceNDArray</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/numba_cuda_DeviceNDArray_as.py#L18 title="def pyarrow_cuda_buffer(nb_arr):
    import pyarrow.cuda as cuda
    ctx = cuda.Context()
    return ctx.buffer_from_object(nb_arr)
">OPTIMAL, FULL</a></td><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/numba_cuda_DeviceNDArray_as.py#L36 title="def cupy_ndarray(nb_arr):
    import cupy
    return cupy.ndarray(nb_arr.shape, dtype=cupy.uint8,
                        strides=nb_arr.strides,
                        memptr=cupy_cuda_MemoryPointer(nb_arr))
">OPTIMAL, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/numba_cuda_DeviceNDArray_as.py#L26 title="def cupy_cuda_MemoryPointer(nb_arr):
    import cupy
    addr = nb_arr.device_ctypes_pointer.value
    size = nb_arr.alloc_size
    mem = cupy.cuda.UnownedMemory(addr, size, nb_arr)
    return cupy.cuda.MemoryPointer(mem, 0)
">OPTIMAL, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/numba_cuda_DeviceNDArray_as.py#L45 title="def xnd_xnd_cuda(nb_arr):
    cbuf = pyarrow_cuda_buffer(nb_arr)
    # DERIVED
    return pyarrow_cuda_buffer_as.xnd_xnd_cuda(cbuf)
">DERIVED, FULL</a></td></tr>
<tr><th>cupy.ndarray</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/cupy_ndarray_as.py#L22 title="def pyarrow_cuda_buffer(cp_arr):
    import pyarrow.cuda as cuda
    ctx = cuda.Context(cp_arr.data.device.id)
    return ctx.foreign_buffer(cp_arr.data.ptr, cp_arr.nbytes)
">OPTIMAL, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/cupy_ndarray_as.py#L15 title="def numba_cuda_DeviceNDArray(cp_arr):
    import numba.cuda as nb_cuda
    return nb_cuda.as_cuda_array(cp_arr)
">OPTIMAL, FULL</a></td><td></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/cupy_ndarray_as.py#L30 title="def cupy_cuda_MemoryPointer(cp_arr):
    return cp_arr.data
">OPTIMAL, FULL</a></td><td>NOT IMPL</td></tr>
<tr><th>cupy MemoryPointer</th><td>NOT IMPL</td><td>NOT IMPL</td><td>NOT IMPL</td><td></td><td>NOT IMPL</td></tr>
<tr><th>xnd.xnd CUDA</th><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/xnd_xnd_cuda_as.py#L18 title="def pyarrow_cuda_buffer(xd_arr):
    import pyarrow as pa
    import pyarrow.cuda as cuda
    buf = pa.py_buffer(memoryview(xd_arr))
    ctx = cuda.Context()
    return ctx.foreign_buffer(buf.address, buf.size)
">OPTIMAL, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/xnd_xnd_cuda_as.py#L46 title="def numba_cuda_DeviceNDArray(xd_arr):
    cbuf = pyarrow_cuda_buffer(xd_arr)
    # DERIVED
    return pyarrow_cuda_buffer_as.numba_cuda_DeviceNDArray(cbuf)
">DERIVED, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/xnd_xnd_cuda_as.py#L38 title="def cupy_ndarray(xd_arr):
    import cupy
    return cupy.ndarray(xd_arr.type.datasize, dtype=cupy.uint8,
                        memptr=cupy_cuda_MemoryPointer(xd_arr))
">OPTIMAL, FULL</a></td><td><a href=https://github.com/plures/arrayviews/blob/master/arrayviews/cuda/xnd_xnd_cuda_as.py#L28 title="def cupy_cuda_MemoryPointer(xd_arr):
    import cupy
    import pyarrow as pa
    buf = pa.py_buffer(memoryview(xd_arr))
    mem = cupy.cuda.UnownedMemory(buf.address, buf.size, xd_arr)
    return cupy.cuda.MemoryPointer(mem, 0)
">OPTIMAL, FULL</a></td><td></td></tr>
</table>
<!--END arrayviews.cuda-support_kernel TABLE-->

### Benchmark: creating array views - CUDA device memory

<!--START arrayviews.cuda-measure_kernel TABLE-->
<table style="width:100%">
<tr><th rowspan=2>Objects</th><th colspan="5">Views</th></tr>
<tr><th>pyarrow CudaBuffer</th><th>numba DeviceNDArray</th><th>cupy.ndarray</th><th>cupy MemoryPointer</th><th>xnd.xnd CUDA</th></tr>
<tr><th>pyarrow CudaBuffer</th><td>1.0</td><td>371.34</td><td>40.55</td><td>27.76</td><td>32.52</td></tr>
<tr><th>numba DeviceNDArray</th><td>82.32</td><td>1.01</td><td>39.64</td><td>23.62</td><td>128.18</td></tr>
<tr><th>cupy.ndarray</th><td>27.41</td><td>361.54</td><td>0.98</td><td>1.16</td><td>NOT IMPL</td></tr>
<tr><th>cupy MemoryPointer</th><td>NOT IMPL</td><td>NOT IMPL</td><td>NOT IMPL</td><td>0.98</td><td>NOT IMPL</td></tr>
<tr><th>xnd.xnd CUDA</th><td>33.79</td><td>428.29</td><td>41.05</td><td>26.1</td><td>1.01</td></tr>
</table>
<!--END arrayviews.cuda-measure_kernel TABLE-->

#### Comments

1. Test arrays are 8-bit unsigned integer arrays of size 51200.
