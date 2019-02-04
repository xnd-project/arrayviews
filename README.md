# ArrayViews

## Matrix of supported array views - host memory

The following table summarizes the support of creating a specific array
view (left-hand-side column) of given array storage
object (top-row). The table cells contains two answers
corresponding to if the view can be created from an array object without or with nulls, respectively.
In the case of `numpy.ndarray`, the `nan` values are interpreted as nulls.

<table style="width:100%">
  <tr>
    <th>Views</th>
    <th colspan="4">Objects</th>
  </tr>
  <tr>
    <th></th>
    <th>numpy.ndarray</th> 
    <th>pandas.Series</th>
    <th>pyarrow.Array</th>
    <th>xnd.xnd</th>
  </tr>
  <tr>
    <th>numpy.ndarray </th>
    <td title="use self"></td> 
    <td>yes/yes</td>
    <td>yes/no</td>
    <td title="np.frombuffer(memoryview(xd_arr), dtype=str(xd_arr.dtype))">yes/no</td>
  </tr>
  <tr>
    <th>pandas.Series</th>
    <td>yes/yes</td> 
    <td></td>
    <td>yes/yes</td>
    <td title=" pd.Series(memoryview(xd_arr), dtype=str(xd_arr.dtype))">yes/no</td>
  </tr>
  <tr>
    <th>pyarrow.Array</th>
    <td>yes/yes</td> 
    <td>yes/yes</td>
    <td></td>
    <td title="pa.Array.from_buffers(pa.from_numpy_dtype(str(xd_arr.dtype)),
            xd_arr.type.datasize//xd_arr.type.itemsize,
            [None, pa_buf])">yes/no</td>
  </tr>
  <tr>
    <th>xnd.xnd</th>
    <td>yes/no</td> 
    <td>yes/no</td>
    <td>yes/no</td>
    <td></td>
  </tr>
</table>

For the implementation of view constructions, see `arrayviews` package source code.

## Matrix of supported array views - CUDA device memory

TODO
