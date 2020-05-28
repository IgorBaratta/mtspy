import numpy
from scipy import sparse
import pytest
import mtspy_cpp


def mat_vec(A, b):
    rows, cols = A.shape
    nnz = A.nnz
    dtype = b.dtype

    if dtype == numpy.float64:
        op = mtspy_cpp.mat_vec_d
    elif dtype == numpy.float32:
        op = mtspy_cpp.mat_vec_f
    elif dtype == numpy.complex64:
        op = mtspy_cpp.mat_vec_cf
    elif dtype == numpy.complex128:
        op = mtspy_cpp.mat_vec_cd

    x = op(rows, cols, nnz,
           A.data, A.indptr,
           A.indices, b)
    return x


N = 1000
v0 = numpy.ones(N, dtype=numpy.float64)
M = sparse.random(N, N, density=0.1, format="csr", dtype=numpy.float64)
v1 = mat_vec(M, v0)
v2 = M@v0
print(numpy.allclose(v1, v2))
