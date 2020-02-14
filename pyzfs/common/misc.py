import sys
import numpy as np
from collections import OrderedDict
import re
from ase import Atoms
from six import string_types
from .units import bohr_to_angstrom


# Miscellaneous functions


def parse_sys_argv(flags=None):
    """Parse sys.argv into a list of unnamed arguments and a dict of named arguments

    It is assumed that a flag starts with "--", and is followed by a certain number
    of values (can be 0)
    Args:
        flags: a dict of flag name: nvalue (how many values are associated with that flag),
            default to None, in which case all elements in sys.argv starting with "--" are
            interpreted as flags with nvalue = 1

    Returns:
        args (list): a list representing unnamed arguments
        kwargs (dict): a dict of named arguments. If nvalue == 0, then the dict value is True;
            if nvalue == 1, the dict value is the following element in sys.argv; if nvalue > 1,
            the dict value is a list of the following nvalue elements in sys.argv.

        elements of args and values of kwargs are always strings, no type conversions
        are performed here

    """

    argv = np.array(sys.argv[1:], dtype=object)
    argc = len(argv)
    unparsed = np.ones(argc, dtype=bool)

    if flags is None:
        flags = {
            flag[2:]: 1
            for flag in argv if "--" in flag
        }

    kwargs = OrderedDict()
    for flag, nvalue in flags.items():
        indices = [i for i in range(argc) if argv[i] == "--{}".format(flag)]
        if len(indices) == 0:
            continue
        elif len(indices) == 1:
            index = indices[0]
            try:
                if nvalue == 0:
                    kwargs[flag] = True
                elif nvalue == 1:
                    kwargs[flag] = argv[index + 1]
                else:
                    kwargs[flag] = argv[index + 1: index + nvalue + 1]
            except IndexError:
                print("cannot parse enough values for flag --{}".format(flag))
            unparsed[index: index + nvalue + 1] = False
        else:
            raise ValueError("flag --{} appeared multiple times".format(flag))

    args = argv[unparsed]

    return args, kwargs


def regex(dtype):
    """Returns the regular expression required by re package

    Args:
        dtype: type of value wanted. int, float or str

    Returns:
        string of regular expression
    """

    if dtype is int:
        return r"-*\d+"
    elif dtype is float:
        return r"-*\d+\.\d*[DeEe]*[+-]*\d*"
    elif dtype is str:
        return r".*"
    else:
        raise ValueError("unsupported type")


def parse_one_value(dtype, content, index=0):
    """Parse one value of type dtype from content

    Args:
        dtype: type of value wanted
        content: a string to be parsed
        index: index of parsed value

    Returns:
        first (if index not specified) value found in content
    """

    results = re.findall(regex(dtype), content)
    if results:
        return dtype(results[index])


def parse_many_values(n, dtype, content):
    """Parse n values of type dtype from content

    Args:
        n: # of values wanted
        dtype: type of values wanted
        content: a string or a list of strings,
                 it is assumed that n values exist in continues
                 lines of content starting from the first line

    Returns:
        a list of n values
    """

    if isinstance(content, string_types) or isinstance(content, np.string_):
        results = re.findall(regex(dtype), content)
        return [dtype(value) for value in results[0:n]]

    results = list()
    started = False
    for i in range(len(content)):
        found = re.findall(regex(dtype), content[i])
        if found:
            started = True
        else:
            if started:
                raise ValueError("cannot parse {} {} variables in content {}".format(
                    n, dtype, content
                ))
        results.extend(found)
        assert len(results) <= n
        if len(results) == n:
            return [dtype(result) for result in results]


def empty_ase_cell(a, b, c, unit="angstrom"):
    if unit == "angstrom":
        s = 1
    elif unit == "bohr":
        s = bohr_to_angstrom
    else:
        raise ValueError
    return Atoms(cell=s * np.array([a, b, c]), pbc=[1, 1, 1])

