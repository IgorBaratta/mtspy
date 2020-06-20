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
