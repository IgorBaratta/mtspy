import mtspy
import numpy
from scipy import sparse
import pytest

dtype_list = [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]


@pytest.mark.parametrize('dtype', dtype_list)
def test_linear_operator(dtype):
    N = 1000
    v0 = numpy.random.rand(N, 1).astype(dtype)
    M = sparse.random(N, N, density=0.1, format="csr", dtype=dtype)

    L = mtspy.aslinearoperator(M)

    v1 = L @ v0
    v2 = M @ v0
    assert(numpy.allclose(v1, v2))

    LL = L @ L
    MM = M @ M
    assert((MM - LL).data.size == 0)

    B = numpy.random.rand(N, 100).astype(dtype)
    LB = L @ B
    MB = M @ B
    assert(numpy.allclose(MB, LB))


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
