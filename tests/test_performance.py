import numpy
from scipy import sparse
from mtspy import thread_control, matvec, matmat


numpy.seterr(all='warn', over='raise')
N = 10000
Ncols = 1000
diags = numpy.arange(start=-200, stop=200)
values = numpy.random.rand(diags.size)
M = sparse.diags(values, diags, shape=(N, N), format='csr', dtype=numpy.float64)


v0 = numpy.ones((N, Ncols), dtype=numpy.float64)

# Set maximum number of threads (tentative) to 4,
# equivalent of setting OMP_NUM_THREADS
thread_control.set_num_threads(4)
GFLOPS = [0., 0., 0.]

# Benchmark scipy/numpy
with thread_control(2, timer=True) as th:
    v2 = M @ v0

GFLOPS[0] = (2 * M.nnz * Ncols / th.elapsed_time) * 1e-9

# Use only one thread locally if the function within the context
# uses threads.
with thread_control(2, timer=True) as th:
    v1 = matmat(M, v0, use_eigen=True)

GFLOPS[1] = (2 * M.nnz * Ncols / th.elapsed_time) * 1e-9

# Use 4 threads locally if the function within the context
# uses threads.
with thread_control(2, timer=True) as th:
    y = matmat(M, v0)

GFLOPS[2] = (2 * M.nnz * Ncols / th.elapsed_time) * 1e-9

assert (numpy.allclose(y - v1, 0))
print(thread_control.get_max_threads())

print(GFLOPS)
