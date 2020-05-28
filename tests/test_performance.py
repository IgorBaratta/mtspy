import numpy
from scipy import sparse
from mtspy import thread_control
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
M = sparse.random(N, N, density=0.8, format="csr", dtype=numpy.float64)

# Set maximum number of threads (tentative),
# equivalent of setting OMP_NUM_THREADS
thread_control.set_num_threads(4)

t0 = time.perf_counter()
v2 = M@v0
t1 = time.perf_counter()
print("Elapsed time (s): ", t1 - t0)

# Use 4 threads locally if the function within the context
# uses threads.
with thread_control(4, timer=True):
    v1 = mat_vec(M, v0)

# Use only one thread locally if the function within the context
# uses threads.
with thread_control(1, timer=True):
    v1 = mat_vec(M, v0)

print(thread_control.get_max_threads())
