import numpy
from scipy import sparse
from mtspy import thread_control, matvec, matmat
import time


numpy.seterr(all='warn', over='raise')
N = 100000
diags = numpy.arange(start = -200, stop=200)
values = numpy.random.rand(diags.size)
M = sparse.diags(values, diags, shape=(N,N), format='csr')


v0 = numpy.ones((N, 200), dtype=numpy.float64)

# Set maximum number of threads (tentative) to 4,
# equivalent of setting OMP_NUM_THREADS
thread_control.set_num_threads(4)

t0 = time.perf_counter()
v2 = M@v0
t1 = time.perf_counter()
print("Elapsed time (s): ", t1 - t0)

# Use only one thread locally if the function within the context
# uses threads.
with thread_control(1, timer=True) as th:
    v1 = matmat(M, v0)

# Use 4 threads locally if the function within the context
# uses threads.
with thread_control(2, timer=True):
    v1 = matmat(M, v0)


assert (numpy.allclose(v1-v2, 0))
print(thread_control.get_max_threads())
