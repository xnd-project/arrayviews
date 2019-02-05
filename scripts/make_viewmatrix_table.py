"""Generate view matrix of array storage objects.
"""
# Author: Pearu Peterson
# Created: February 2019

import inspect


def get_result(source):
    tags = []
    if 'get_bitmap' in source:
        tags.append('GENBITMAP')
    else:
        tags.append('OPTIMAL')
    if 'NotImplemented' in source:
        tags.append('PARTIAL')
    else:
        tags.append('FULL')
    return ', '.join(tags)


target_name_title = dict(
    numpy_ndarray='numpy.ndarray',
    pandas_series='pandas.Series',
    pyarrow_array='pyarrow.Array',
    xnd_xnd='xnd.xnd',
    pyarrow_cuda_buffer='pyarrow CudaBuffer',
    numba_cuda_DeviceNDArray='numba DeviceNDArray',
    cupy_ndarray='cupy.ndarray',
    cupy_cuda_MemoryPointer='cupy MemoryPointer'
)


def make_viewmatrix_table(package, kernel):
    table_name = package.__name__ + '-' + kernel.__name__

    target_names = []
    for modulename, module in package.__dict__.items():
        if not modulename.endswith('_as'):
            continue
        target_name = modulename.split('.')[-1][:-3]
        target_names.append(target_name)

    lines = []
    lines.append(f'<!--START {table_name} TABLE-->')
    lines.append('<table style="width:100%">')
    lines.append('<tr><th rowspan=2>Objects</th>'
                 f'<th colspan="{len(target_names)}">'
                 'Views</th></tr>')
    row = []
    row.append(f'<tr>')
    for source_name in target_names:
        row.append(f'<th>{target_name_title[source_name]}</th>')
    row.append('</tr>')
    lines.append(''.join(row))

    for source_name in target_names:
        row = []
        row.append(f'<tr><th>{target_name_title[source_name]}</th>')
        for target_name in target_names:
            source_module = getattr(package, source_name + '_as', None)
            row.append(f'<td>{kernel(source_module, target_name)}</td>')
        row.append('</tr>')
        lines.append(''.join(row))
    lines.append('</table>')
    lines.append(f'<!--END {table_name} TABLE-->')

    return '\n'.join(lines)


def support_kernel(source_module, target_name):
    source_name = source_module.__name__.split('.')[-1][:-3]
    if target_name == source_name:
        return ''
    target_func = source_module.__dict__.get(target_name, None)
    if target_func is None:
        return 'NOT IMPL'
    module_path = '/'.join(source_module.__name__.split('.'))
    source_lines, lineno = inspect.getsourcelines(target_func)
    source = ''.join(source_lines)
    result = get_result(source)
    i0 = source.find('"""')
    i1 = source.find('"""', i0+1)
    source = source[:i0].rstrip() + source[i1+3:]
    link = ('https://github.com/plures/arrayviews/blob/master/'
            f'{module_path}.py#L{lineno}')
    return f'<a href={link} title="{source}">{result}</a>'


def measure_kernel(source_module, target_name):
    import timeit

    def dummy_func(obj):
        return obj

    source_name = source_module.__name__.split('.')[-1][:-3]
    if target_name == source_name:
        target_func = dummy_func
    else:
        target_func = source_module.__dict__.get(target_name, None)
    if target_func is None:
        return 'NOT IMPL'

    random = source_module.__dict__.get('random', None)
    if random is None:
        return 'random NOT IMPL'

    number, size = 100000, 51200
    #number, size = 100, 512
    src1 = random(size)
    r1 = timeit.timeit('target_func(obj)', 'target_func(obj)',
                       number=number,
                       globals=dict(target_func=target_func, obj=src1))
    r0 = timeit.timeit('target_func(obj)', 'target_func(obj)',
                       number=number,
                       globals=dict(target_func=dummy_func, obj=src1))
    args = inspect.getfullargspec(random).args
    if 'nulls' in args:
        src2 = random(size, nulls=True)
        try:
            r2 = timeit.timeit('target_func(obj)', number=number,
                               globals=dict(target_func=target_func, obj=src2))
        except NotImplementedError:
            r2 = None
        if r2 is None:
            return f'{round(r1/r0, 2)}(N/A)'
        return f'{round(r1/r0, 2)}({round(r2/r0, 2)})'
    return f'{round(r1/r0, 2)}'


def update_README_md(package, path):
    f = open(path, 'r')
    orig_content = content = f.read()
    f.close()

    for kernel in [support_kernel, measure_kernel]:
        table = make_viewmatrix_table(package, kernel)
        first_line = table.split("\n", 1)[0]
        last_line = table.rsplit("\n", 1)[-1]
        i0 = content.find(first_line)
        i1 = content.find(last_line)
        if i0 >= 0 and i1 > i0:
            content = content[:i0] + table + content[i1+len(last_line):]
        else:
            print(f'Insert `{first_line} {last_line}` to {path} and re-make.')

    if content != orig_content:
        print(f'Updating {path}')
        f = open(path, 'w')
        f.write(content)
        f.close()


if __name__ == '__main__':
    import os
    import arrayviews
    import arrayviews.cuda
    path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
    update_README_md(arrayviews, path=path)
    update_README_md(arrayviews.cuda, path=path)
