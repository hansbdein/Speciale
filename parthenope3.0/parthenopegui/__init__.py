# -*- coding: utf-8 -*-
"""GUI Package for the PArthENoPE BBN code"""

import sys

__author__ = "Stefano Gariazzo, Pablo Fernández de Salas, Ofelia Pisanti"
__email__ = "parthenope@na.infn.it"

__website__ = "http://parthenope.na.infn.it"

__version__ = "3.0.0"
__version_date__ = "10/03/2021"

__all__ = [
    "basic",
    "configuration",
    "errorManager",
    "mainWindow",
    "plotter",
    "plotUtils",
    "runner",
    "runUtils",
    "setrun",
    "texts",
]

if sys.version_info[0] < 3:
    FileNotFoundErrorClass = IOError
else:
    FileNotFoundErrorClass = FileNotFoundError
