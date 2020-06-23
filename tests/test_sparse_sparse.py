import numpy
import mtspy
from scipy.sparse import rand, identity, csr_matrix

N = 500
A = rand(N, N, density=0.01, format="csr", random_state=42)
I = rand(N, N, density=0.01, format="csr", random_state=42)

# small
# A = rand(100, 100, density=0.5, format="csr", random_state=42)
# I = rand(100, 100, density=0.5, format="csr", random_state=42)


print("Starting Computation")

m, n, k = A.shape[0], I.shape[1], A.shape[1]

mtspy.thread_control.set_num_threads(4)

assert(A.shape[1] == I.shape[0])

with mtspy.thread_control(1, True):
    C1 = A @ I

with mtspy.thread_control(1, True) as th:
    [data, indptr, indices] = mtspy.cpp.sparse_sparse_tf(m, n, k,
                                                         A.data, A.indptr, A.indices,
                                                         I.data, I.indptr, I.indices)

C = csr_matrix((data, indices, indptr), shape=(m, n))

with mtspy.thread_control(1, True) as th:
    mtspy.cpp.sparse_sparse(m, n, k,
                            A.data, A.indptr, A.indices,
                            I.data, I.indptr, I.indices,
                            C.data, C.indptr, C.indices)


assert(numpy.alltrue(C1.indptr == C.indptr))
assert(numpy.alltrue(numpy.sort(C.indices) == numpy.sort(C1.indices)))
assert(numpy.isclose(numpy.linalg.norm(C1.data), numpy.linalg.norm(C.data)))
