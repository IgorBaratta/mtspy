import numpy
import mtspy
from scipy.sparse import rand, identity


A = rand(1000, 1000, density=0.1, format="csr", random_state=42)
I = rand(1000, 1000, density=0.1, format="csr", random_state=42)


with mtspy.thread_control(1, True):
    C1 = A @ I


with mtspy.thread_control(1, True) as th:
    data, indptr, indices = mtspy.cpp.sparse_sparse(A.shape[0], A.data, A.indptr, A.indices,
                                                    I.data, I.indptr, I.indices)


assert(numpy.alltrue(C1.indptr == indptr))
assert(numpy.alltrue(numpy.sort(indices) == numpy.sort(C1.indices)))
