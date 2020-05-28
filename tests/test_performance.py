import numpy
from scipy import sparse
from mtspy import thread_control, matvec
import time

N = 100
v0 = numpy.ones(N, dtype=numpy.float64)
M = sparse.random(N, N, density=0.8, format="csr", dtype=numpy.float64)

# Set maximum number of threads (tentative) to 4,
# equivalent of setting OMP_NUM_THREADS
thread_control.set_num_threads(4)

t0 = time.perf_counter()
v2 = M@v0
t1 = time.perf_counter()
print("Elapsed time (s): ", t1 - t0)

# Use only one thread locally if the function within the context
# uses threads.
with thread_control(1) as th:
    v1 = matvec(M, v0)

# Use 4 threads locally if the function within the context
# uses threads.
with thread_control(4, timer=True):
    v1 = matvec(M, v0)

print(thread_control.get_max_threads())
