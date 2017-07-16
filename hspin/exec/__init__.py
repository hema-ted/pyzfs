import sys
from collections import OrderedDict


def parse_sys_argv():
    argv_iter = iter(sys.argv[1:])  # Exclude 0th argument which always contains script name
    kwargs = OrderedDict()
    for (key, value) in zip(argv_iter, argv_iter):
        kwargs[key.partition("--")[2]] = value
    return kwargs
