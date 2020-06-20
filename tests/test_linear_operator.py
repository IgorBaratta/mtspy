import mtspy
import numpy
from scipy import sparse
import pytest

dtype_list = [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]
matrix_type = ["csr", "csc"]


@pytest.mark.parametrize('dtype', dtype_list)
@pytest.mark.parametrize('mtype', matrix_type)
def test_linear_operator(dtype, mtype):
    N = 1000
    v0 = numpy.random.rand(N, 1).astype(dtype)
    M = sparse.random(N, N, density=0.1, format=mtype, dtype=dtype)

    L = mtspy.aslinearoperator(M)

    v1 = L @ v0
    v2 = L.matvec(v0)
    v3 = M @ v0
    assert(numpy.allclose(v1, v2))
    assert(numpy.allclose(v1, v3))

    C = mtspy.aslinearoperator(L)

    MM = M @ M
    LC = L @ C
    LM = L._matmat(M)

    assert((MM - LC).data.size == 0)
    assert((MM - LM).data.size == 0)

    B = numpy.random.rand(N, 100).astype(dtype)
    assert(numpy.allclose(L @ B, M @ B))


def test_linear_noktype():
    N = 1000
    dense = numpy.ones((N, N))
    try:
        L = mtspy.aslinearoperator(dense)
    except TypeError as err:
        assert (err == TypeError)


if __name__ == "__main__":
    N = 1000
    dtype = numpy.float64
    v0 = numpy.random.rand(N, 1).astype(dtype)
    M = sparse.random(N, N, density=0.1, format="csr", dtype=dtype)

    L = mtspy.aslinearoperator(M)

    v1 = L @ v0
    v2 = M @ v0
    assert(numpy.allclose(v1, v2))

    LL = L @ L
    MM = M @ L
    assert((MM - LL).data.size == 0)

    B = numpy.random.rand(N, 100).astype(dtype)
    LB = L @ B
    MB = M @ B

    assert(numpy.allclose(MB, LB))
