from abc import ABCMeta, abstractmethod

class WavefunctionLoader(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def load(self, indices):
        pass
