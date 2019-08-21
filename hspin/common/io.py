from __future__ import absolute_import, division, print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

class indent:
    """Decorator to make all print commands in a function become indented"""

    def __init__(self, n=0, prefix=""):
        """
        Args:
            n (int): indentation, default to zero
        """
        self.prefix = prefix + " " * n
        self.builtin_print = __builtin__.print
        self.overridden_print = None

    def __call__(self, func):
        def closure(*args, **kwargs):
            self.overridden_print = __builtin__.print
            __builtin__.print = self.indented_print
            func(*args, **kwargs)
            __builtin__.print = self.overridden_print
        return closure

    def indented_print(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            arg = args[0]
            self.builtin_print("{}{}".format(
                self.prefix, str(arg).replace("\n", "\n" + self.prefix)
            ))
        else:
            self.builtin_print(self.prefix, *args, **kwargs)
