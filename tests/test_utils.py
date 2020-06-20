import numpy
import mtspy

from mtspy.utils import get_csr_matrix, IterationCallback, ResidualCallback
from scipy.sparse import identity
from scipy.sparse.linalg import cg


def test_donwload_matrix():
    matrix = get_csr_matrix("HB/1138_bus")
    L = mtspy.aslinearoperator(matrix)

    m, n = L.shape
    nnz = L.matrix.nnz
    assert(m == 1138)
    assert(n == 1138)
    assert(nnz == 4054)

    # Check if matrix market file already exists,
    # reuse file if it is True
    matrix = get_csr_matrix("HB/1138_bus")


def test_callbacks():
    L = mtspy.aslinearoperator(identity(100))
    b = numpy.ones(100)

    itcount = IterationCallback()
    residuals = ResidualCallback()

    x1, info = cg(L, b, callback=itcount)
    assert(info == 0)
    assert(itcount.nit == 1)

    x2, info = cg(L, b, callback=residuals)
    assert(info == 0)
    assert(sum(residuals.residual - b).all() == 0)

    assert (numpy.allclose(x1, x2))
