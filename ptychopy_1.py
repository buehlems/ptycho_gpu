import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs
import p2p
import matplotlib.pylab as plt
import scipy.sparse as aps
from scipy.io import loadmat


def pDiag(s, k):
    k = np.reshape(k, (1, k.shape[0]))
    k = k*s[0]
    I = ((np.asarray(range(0,s[0]*s[1], s[0]+1)).reshape((s[1]),
                     1)+k)) % (s[0]*s[1])
    I[:,(k<0)[0]] = np.sort(I[:,(k<0)[0]], axis = 0)
    return I


def make_sp_cof(l1, d1, gridx, f1, M):
    l = np.repeat((np.asarray(range(1, l1 + 1, 1)) - np.transpose(np.asarray(range(1,
                                                                                   l1 + 1, 1))).reshape((l1, 1)))[:, :,
                  np.newaxis], d1, axis=2)

    d = np.repeat(np.repeat(np.asarray(range(1, d1 + 1, 1))[np.newaxis, :],
                            l1, axis=0)[np.newaxis, :, :], l1, axis=0)
    p = np.repeat(np.minimum(np.tile(range(l1), (l1, 1)),
                             np.transpose(np.tile(range(l1),
                                                  (l1, 1))))[:, :, np.newaxis], d1, axis=2)
    b = np.reshape(f1, (l1, 1, d1), order='f') * np.reshape(f1.conj(), (1, l1, d1), order='f')
    indd = pDiag(np.array([M, M]), np.array(range(l1)))
    indd = indd[gridx, :]
    indd = np.unravel_index(indd, np.array([M, M]))[0]
    q = np.transpose(indd[:, p.reshape((p.size), order='F')])
    p = np.tile(range(np.size(gridx)), (q.shape[0], 1))

    q = q + (l.reshape((l.size, 1), order='F') + l1 - 1) * M
    p = p + (d.reshape((d.size, 1), order='F') - 1) * len(gridx)

    b = np.tile(b.reshape((np.size(b), 1),order='f'), (1, np.size(gridx)))
    a = csr_matrix((b.reshape((b.size),order='f'), (p.reshape((p.size),order='f'), q.reshape((q.size),order='f'))),
                   shape=(np.size(gridx) * d1, M * (2 * l1 - 1)))
    return a


def reconEBAS2d(y, f1, f2, gridx, gridy, mtype):
    symFilters = True
    if f2.shape[0] == 0:
        f2 = f1
    else:
        symFilters = False

    if gridy.shape[0] == 0:
        gridy = gridx
    else:
        symFilters = False
    [l1, d1] = f1.shape
    [l2, d2] = f2.shape
    if mtype == 'periodic':
        M = gridx.max( ) +1
        N = gridy.max( ) +1
    elif mtype == 'zeropadding':
        M = l1- 1 + gridx.max()
        N = l2 - 1 + gridy.max()
    else:
        print('error')
    a = make_sp_cof(l1, d1, gridx, f1, M)
    y = np.reshape(y, (gridx.size * d1, gridy.size * d2), order='F')
    # replace the spsolve with p2p
    a_c = np.zeros((2 * len(a.data)))
    a_c[0::2] = np.real(a.data)
    a_c[1::2] = np.imag(a.data)
    p2p.py2petsc(a_c, a.indices, a.indptr, y)
    # replace the spsolve with p2p
    x = spsolve(a, y)
    if symFilters == False:
        a = make_sp_cof(l2, d2, gridy, f2, N)
    x = spsolve(a, np.transpose(x)) / 2
    x = np.reshape(x, (M, (2 * l1 - 1), N, (2 * l2 - 1)), order='F')
    x = x + np.conj(x[:, ::-1, :, ::-1])
    x_amp = np.sqrt(x[:, l1, :, l2])

    x[abs(x) == 0] = 1
    x = x / np.abs(x)
    indd0 = pDiag(np.array([M, M]), np.array(range(1 - l1, l1)))
    indd0 = np.unravel_index(indd0, np.array([M, M]))

    indd1 = pDiag(np.array([N, N]), np.array(range(1 - l2, l2)))
    indd1 = np.unravel_index(indd1, np.array([N, N]))

    indd1_0 = np.asarray(indd1[0])
    indd1_1 = np.asarray(indd1[1])
    indd1_0 = (indd1_0) * M
    indd1_1 = (indd1_1) * M

    i0 = np.reshape(indd0[0], (np.size(indd0[0]), 1),
                    order='F') + np.reshape(indd1_0, (1, indd1_0.size), order='F')
    i1 = np.reshape(indd0[1], (np.size(indd0[1]), 1),
                    order='F') + np.reshape(indd1_1, (1, indd1_1.size), order='F')

    x = csr_matrix((x.reshape((x.size),order='f'), (i0.reshape((i0.size),order='f'), i1.reshape((i1.size),order='f'))),
                   shape=(M * N, M * N))
    u = np.asarray(eigs(x, 1)[1])
    u[u == 0] = 1
    u = u / np.abs(u)
    x = x_amp * np.reshape(u, (M, N), order='F')

    return x