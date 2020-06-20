# The matrix ML_Laplace has been obtained by discretizing a 2D
# Poisson equation with a Meshless Local Petrov-Galerkin method.

from scipy.sparse.linalg import LinearOperator, gmres, cg
import numpy

from mtspy.utils import get_csr_matrix, IterationCallback
import mtspy

matrix = mtspy.utils.get_csr_matrix("ACUSIM/Pres_Poisson")


A = mtspy.aslinearoperator(matrix)
b = numpy.random.rand(A.shape[0])


# with mtspy.thread_control(2, True):
#     itcount = IterationCallback()
#     x = cg(matrix, b, callback=itcount)

# print(itcount.nit)

# with mtspy.thread_control(2, True):
#     itcount = IterationCallback()
#     x = cg(L, b, callback=itcount)

# print(itcount.nit)
