from distutils.core import setup, Extension

PETSC_ROOT='/home/gpu28/opt/petsc'

module1 = Extension('p2p',
                    sources = ['py2petsc.c','ptycho_petsc.c'],
                    include_dirs=[PETSC_ROOT+'/include'],
                    library_dirs=[PETSC_ROOT+'/lib'],
                    libraries=['petsc'],
                    extra_compile_args = ['-Wl,-rpath,'+PETSC_ROOT+'/lib']
                    )

setup (name = 'p2p',
       version = '1.0',
       description = 'This is a package to call PetSC from Python',
       ext_modules = [module1])
