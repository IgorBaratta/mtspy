import numpy
from scipy import sparse
from mtspy import thread_control, matvec
import time

N = 1000
v0 = numpy.ones(N, dtype=numpy.float64)
diags = numpy.arange(start = -300, stop=300)
values = numpy.random.rand(diags.size)
M = sparse.diags(values, diags, shape=(N,N), format='csr')
# M = sparse.identity(N).tocsr()

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
    v1 = matvec(M, v0)

# Use 4 threads locally if the function within the context
# uses threads.
with thread_control(4, timer=True):
    v1 = matvec(M, v0)

print(thread_control.get_max_threads())
