import ptychopy_1
from ptychopy_1 import make_sp_cof, filters
from scipy.io import loadmat
import numpy as np
import p2p as p2p

mtx = loadmat('Python_testmatrix.mat')

B = mtx['Y']
f1 = filters('Ptycho',4,5)
[l1, d1] = f1.shape
f2 = np.array([])
gridx = np.array(range(64))
gridy = np.array([])
mtype = 'periodic'

A = make_sp_cof(l1, d1, gridx, f1, gridx.max()+1)
X = np.zeros([A.shape[1],2*B.shape[1]])

A_data = np.zeros((2 * len(A.data)))
A_data[0::2] = np.real(A.data)
A_data[1::2] = np.imag(A.data)

p2p.py2petsc(A_data,A.indices,A.indptr,B,X)
