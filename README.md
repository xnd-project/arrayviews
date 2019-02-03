# ArrayViews

## Matrix of supported array views - host memory

The following table summarizes the support of creating an array
storage view (in left-hand-side column) of some other array storage
object (in top-column). The table cells contains two answers
corresponding to array objects without or with nulls. In the case of
`numpy.ndarray`, the `nan` values are interpreted as nulls.
```
|                     | numpy.ndarray object | pandas.Series object | pyarrow.Array object | xnd.xnd object |
| numpy.ndarray view: |                      | yes / yes            | yes / no             | yes / no       |
| pandas.Series view: | yes / yes            |                      | yes / yes            | yes / no       |
| pyarrow.Array view: | yes / yes            | yes / yes            |                      | yes / no       |
| xnd.xnd view:       | yes / no             | yes / no             | yes / no             |                |
```
For the implementation of view constructions, see `arrayviews` package source code.

## Matrix of supported array views - CUDA device memory

TODO
