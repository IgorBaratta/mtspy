import scipy.sparse as sp
import numpy

from mtspy.sparse_ops import matvec, matmat, spmatmat


class LinearOperator(sp.linalg.LinearOperator):
    def __init__(self, matrix: sp.csr_matrix):
        self.matrix = matrix
        super().__init__(shape=matrix.shape, dtype=matrix.dtype)
        self.__adj = None
        self.args = (matrix,)

    def _matvec(self, x):
        return matvec(self.matrix, x)

    def _matmat(self, X):
        if isinstance(X, LinearOperator):
            return spmatmat(self.matrix, X.matrix)
        elif isinstance(X, sp.csr_matrix):
            return spmatmat(self.matrix, X)
        elif isinstance(X, numpy.ndarray):
            return matmat(self.matrix, X)
        else:
            raise TypeError("type not understood")


def aslinearoperator(A):
    """
    Return A as a mtspy LinearOperator.
    """
    # check type
    if isinstance(A, LinearOperator):
        return A
    elif isinstance(A, sp.spmatrix):
        return LinearOperator(A.tocsr())
    else:
        try:
            return LinearOperator(sp.csr_matrix(A))
        except TypeError:
            raise TypeError('type not understood')
