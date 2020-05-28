import numpy
from scipy import sparse
import mtspy_cpp
import time


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


N = 10000
v0 = numpy.ones(N, dtype=numpy.float64)
M = sparse.random(N, N, density=0.1, format="csr", dtype=numpy.float64)

t0 = time.perf_counter()
v1 = mat_vec(M, v0)
t1 = time.perf_counter()
print(t1 - t0, "seconds")

t0 = time.perf_counter()
v2 = M@v0
t1 = time.perf_counter()
print(t1 - t0, "seconds")

print(numpy.allclose(v1, v2))
