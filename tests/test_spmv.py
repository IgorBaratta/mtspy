import numpy
from scipy import sparse
import pytest

import mtspy

dtype_list = [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]
@pytest.mark.parametrize('dtype', dtype_list)
def test_spmv(dtype):
    N = 1000
    v0 = numpy.ones(N, dtype=dtype)
    M = sparse.random(N, N, density=0.1, format="csr", dtype=dtype)
    v1 = mtspy.matvec(M, v0)
    v2 = M@v0
    assert(numpy.allclose(v1, v2))

