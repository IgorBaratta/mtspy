import numpy
from scipy import sparse
import mtspy_cpp


def mat_vec(A, b):
    rows, cols = A.shape
    nnz = A.nnz
    x = mtspy_cpp.mat_vec_d(rows, cols, nnz,
                            A.data, A.indptr,
                            A.indices, b)
    return x


N = 1000
v0 = numpy.ones(N, dtype=numpy.float64)
M = sparse.random(N, N, density=0.1, format="csr", dtype=numpy.float64)

v1 = mat_vec(M, v0)
v2 = M@v0

assert(numpy.allclose(v1, v2))
