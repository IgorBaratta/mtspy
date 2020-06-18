# The matrix ML_Laplace has been obtained by discretizing a 2D
# Poisson equation with a Meshless Local Petrov-Galerkin method.

from scipy.sparse.linalg import LinearOperator, gmres
import numpy

from mtspy.utils import get_matrix, IterationCallback
import mtspy

matrix = mtspy.utils.get_matrix("Janna/ML_Laplace")


class PoissonOperator(LinearOperator):
    def __init__(self, matrix):
        self.matrix = matrix
        super().__init__(shape=matrix.shape, dtype=matrix.dtype)

    def _matvec(self, x):
        return mtspy.matvec(self.matrix, x)

    def _matmat(self, x):
        return mtspy.matmat(self.matrix, x)


itcount = IterationCallback()

L = PoissonOperator(matrix)
b = numpy.random.rand(L.shape[0])

with mtspy.thread_control(1, True):
    x = gmres(matrix, b, maxiter=200, callback=itcount)


with mtspy.thread_control(2, True):
    x = gmres(L, b, maxiter=200, callback=itcount)
