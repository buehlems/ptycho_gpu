init.src: source this file to properly setup Python on Taurus
build_p2p.sh: build the python to PetSC module
create_test_matrices.py: test program in Python
ptychopy_1.py: needed by create_test_matrices.py
py2petsc.c: the C source code
setup_p2p.py: needed by build_p2p.sh

Extended by MPI. Run with slurm e.g.:
salloc -x taurusi2108 -p gpu2-interactive --gres=gpu:1 --time=00:10:00
srun python create_test_matrices.py
