# ArrayViews

## Matrix of supported array views - host memory

The following table summarizes the support of creating an array
storage view (in left-hand-side column) of some other array storage
object (in top-column). The table cells contains two answers
corresponding to array objects without or with nulls. In the case of
`numpy.ndarray`, the `nan` values are interpreted as nulls.

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
    <td></td> 
    <td>yes/yes</td>
    <td>yes/no</td>
    <td>yes/no</td>
  </tr>
  <tr>
    <th>pandas.Series</th>
    <td>yes/yes</td> 
    <td></td>
    <td>yes/yes</td>
    <td>yes/no</td>
  </tr>
  <tr>
    <th>pyarrow.Array</th>
    <td>yes/yes</td> 
    <td>yes/yes</td>
    <td></td>
    <td>yes/no</td>
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
