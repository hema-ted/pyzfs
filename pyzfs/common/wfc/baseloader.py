from __future__ import absolute_import, division, print_function

import os
from abc import ABCMeta, abstractmethod
from pprint import pprint

from ..parallel import mpiroot


class WavefunctionLoader(object):
    __metaclass__ = ABCMeta

    def __init__(self, memory="critical"):
        self.wfc = None
        self.memory = memory
        self.scan()
        self.info()

    @abstractmethod
    def scan(self):
        """Scan current directory, construct wavefunction object"""
        if mpiroot:
            print("\n{}: scanning current working directory \"{}\"...\n".format(
                self.__class__.__name__, os.getcwd()
            ))

    @abstractmethod
    def load(self, iorbs, sdm):
        """Load read space KS orbitals to memory, store in wfc.iorb_psir_map."""
        if mpiroot:
            print("\n{}: loading orbitals into memory... (memory mode: \"{}\")\n".format(
                self.__class__.__name__, self.memory
            ))

    def info(self):
        if mpiroot:
            wfc = self.wfc
            print("   nuwfcs = {}, ndwfcs = {}, nwfcs = {}".format(
                wfc.nuorbs, wfc.ndorbs, wfc.norbs
            ))
            for iorb in range(wfc.norbs):
                print("     spin = {}     band = {}     file = {}".format(
                    wfc.iorb_sb_map[iorb][0], wfc.iorb_sb_map[iorb][1], wfc.iorb_fname_map[iorb]
                ))
            print("\nSystem Overview:")
            print("  Cell: ")
            pprint(wfc.cell.__dict__, indent=4)
            print("  FFT Grid: ")
            pprint(wfc.ft.__dict__, indent=4)
