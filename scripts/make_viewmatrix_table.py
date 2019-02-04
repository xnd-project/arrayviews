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


def make_viewmatrix_table(package):

    target_names = []
    for modulename, module in package.__dict__.items():
        if not modulename.endswith('_as'):
            continue
        target_name = modulename.split('.')[-1][:-3]
        target_names.append(target_name)

    lines = []
    lines.append(f'<!--START {package.__name__} TABLE-->')
    lines.append('<table style="width:100%">')
    lines.append(f'<tr><th>Views</th><th colspan="{len(target_names)}">'
                 'Objects</th></tr>')
    row = []
    row.append(f'<tr><th></th>')
    for source_name in target_names:
        row.append(f'<th>{source_name}</th>')
    row.append('</tr>')
    lines.append(''.join(row))

    for target_name in target_names:
        row = []
        row.append(f'<tr><th>{target_name}</th>')
        for source_name in target_names:
            source_module = getattr(package, target_name + '_as', None)
            module_path = '/'.join(source_module.__name__.split('.'))
            func = source_module.__dict__.get(source_name, None)
            if target_name == source_name:
                row.append('<td></td>')
            elif func is None:
                row.append('<td>NOT IMPL</td>')
            else:
                source_lines, lineno = inspect.getsourcelines(func)
                source = ''.join(source_lines)
                result = get_result(source)
                i0 = source.find('"""')
                i1 = source.find('"""', i0+1)
                source = source[:i0].rstrip() + source[i1+3:]
                link = ('https://github.com/plures/arrayviews/blob/master/'
                        f'{module_path}.py#L{lineno}')
                row.append(
                    f'<td><a href={link} title="{source}">{result}</a></td>')
        row.append('</tr>')
        lines.append(''.join(row))
    lines.append('</table>')
    lines.append(f'<!--END {package.__name__} TABLE-->')
    return '\n'.join(lines)


def update_README_md(package, path):
    table = make_viewmatrix_table(arrayviews)
    first_line = table.split("\n", 1)[0]
    last_line = table.rsplit("\n", 1)[-1]
    f = open(path, 'r')
    content = f.read()
    f.close()

    i0 = content.find(first_line)
    i1 = content.find(last_line)

    if i0 >= 0 and i1 > i0:
        new_content = content[:i0] + table + content[i1+len(last_line):]
        print(f'Updating {path}')
        f = open(path, 'w')
        f.write(new_content)
        f.close()
    else:
        print(f'Insert `{first_line} {last_line}` to {path} and re-make.')


if __name__ == '__main__':
    import os
    import arrayviews
    update_README_md(arrayviews, path=os.path.join(os.path.dirname(__file__),
                                                   '..', 'README.md'))
