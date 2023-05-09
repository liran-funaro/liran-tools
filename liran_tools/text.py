"""
Author: Liran Funaro <liran.funaro@gmail.com>

Copyright (C) 2006-2023 Liran Funaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


def try_basic_type(val):
    try:
        return int(val)
    except:
        pass

    try:
        return float(val)
    except:
        pass

    try:
        return dict(true=True, false=False)[val.lower().strip()]
    except:
        pass

    try:
        return int(val.replace(",", ""))
    except:
        pass

    return val


mem_size_factor = {
    'g': 2**30, 'gb': 2**30,
    'm': 2**20, 'mb': 2**20,
    'k': 2**10, 'kb': 2**10,
}


def str_to_mem_size(sz_str: str, sz_scale: str):
    return int(round(float(sz_str) * mem_size_factor.get(sz_scale.lower(), 1), 0))


def mem_size_str(sz: float):
    if sz >= 2**30:
        return f'{round(sz / 2 ** 30, 2)}GB'
    elif sz >= 2**20:
        return f'{round(sz / 2 ** 20, 2)}MB'
    elif sz >= 2**10:
        return f'{round(sz / 2 ** 10, 2)}KB'
    else:
        return f'{round(sz, 0):g}B'


def large_size_str(size):
    if size is None:
        return "?"
    for s, k in ((1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K')):
        if size >= s:
            return f"{size / s:.0f}{k}"
    return f"{size:.0f}"
