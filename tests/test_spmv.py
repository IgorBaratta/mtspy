import numpy
from scipy import sparse
import pytest
import mtspy


def mat_vec(A, b):
    rows, cols = A.shape
    nnz = A.nnz
    dtype = b.dtype

    if dtype == numpy.float64:
        op = mtspy.mat_vec_d
    elif dtype == numpy.float32:
        op = mtspy.mat_vec_f
    elif dtype == numpy.complex64:
        op = mtspy.mat_vec_cf
    elif dtype == numpy.complex128:
        op = mtspy.mat_vec_cd

    x = op(rows, cols, nnz,
           A.data, A.indptr,
           A.indices, b)
    return x

dtype_list = [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]
@pytest.mark.parametrize('dtype', dtype_list)
def test_spmv(dtype):
    N = 1000
    v0 = numpy.ones(N, dtype=dtype)
    M = sparse.random(N, N, density=0.1, format="csr", dtype=dtype)
    v1 = mat_vec(M, v0)
    v2 = M@v0
    assert(numpy.allclose(v1, v2))
