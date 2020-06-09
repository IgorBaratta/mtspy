import numpy
from mtspy_cpp import matmat as mtm
from scipy import sparse
from mtspy import thread_control, matvec, matmat
import time


numpy.seterr(all='warn', over='raise')
N = 100000
diags = numpy.arange(start=-20, stop=20)
values = numpy.random.rand(diags.size)
M = sparse.diags(values, diags, shape=(N, N), format='csr', dtype=numpy.float64)


v0 = numpy.ones((N, 1000), dtype=numpy.float64)

# Set maximum number of threads (tentative) to 4,
# equivalent of setting OMP_NUM_THREADS
thread_control.set_num_threads(4)

# Benchmark scipy/numpy
with thread_control(1, timer=True):
    v2 = M @ v0


# Use only one thread locally if the function within the context
# uses threads.
with thread_control(1, timer=True):
    v1 = matmat(M, v0)

# Use 4 threads locally if the function within the context
# uses threads.
with thread_control(2, timer=True):
    y = mtm(M.shape[0], M.shape[1], M.nnz,
            M.data, M.indptr,
            M.indices, v0)


assert (numpy.allclose(y - v1, 0))
print(thread_control.get_max_threads())
