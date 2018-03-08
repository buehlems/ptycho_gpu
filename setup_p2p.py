from distutils.core import setup, Extension
import numpy as np

module1 = Extension('p2p',
                    sources = ['py2petsc.c'])

setup (name = 'p2p',
       version = '1.0',
       description = 'This is a package to call PetSC from Python',
       ext_modules = [module1],
       include_dirs = [np.get_include()])

