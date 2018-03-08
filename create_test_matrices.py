import ptychopy_1
from ptychopy_1 import make_sp_cof
import numpy as np
import p2p as p2p

B = np.reshape(np.asarray(range(16))*0.5,[4,4])
A = make_sp_cof(2,3,range(2),0.5*np.reshape(range(1,7),[2,3]),2)
X = np.zeros([A.shape[1],B.shape[1]])
print("types A=",type(A),"types A.data=",type(A.data),"types A.indices=",type(A.indices),"types A.indptr=",type(A.indptr),"types B=",type(B));
print("types A.data[0]=",type(A.data[0]),"types A.indices[0]=",type(A.indices[0]),"types A.indptr[0]=",type(A.indptr[0]),"types B[0][0]=",type(B[0][0]));
p2p.py2petsc(A.data,A.indices,A.indptr,B,X)

# print(A)
print("A.data=",A.data)
print("A.indices=",A.indices)
print("A.iptr=",A.indptr)
print("B=",B)


