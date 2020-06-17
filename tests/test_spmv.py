import numpy
from scipy import sparse
import pytest

import mtspy

dtype_list = [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]
if mtspy.cpp.has_eigen():
    eigen_backend = [False, True]
else:
    eigen_backend = [False]


@pytest.mark.parametrize('dtype', dtype_list)
@pytest.mark.parametrize('use_eigen', eigen_backend)
def test_spmv(dtype, use_eigen):
    N = 1000
    v0 = numpy.random.rand(N, 1).astype(dtype)
    M = sparse.random(N, N, density=0.1, format="csr", dtype=dtype)
    v1 = mtspy.matvec(M, v0, use_eigen)
    v2 = M @ v0
    assert(numpy.allclose(v1, v2))


@pytest.mark.parametrize('dtype', dtype_list)
@pytest.mark.parametrize('use_eigen', eigen_backend)
def test_spmm(dtype, use_eigen):
    N = 1000
    v0 = numpy.ones((N, 10), dtype=dtype)
    M = sparse.random(N, N, density=0.1, format="csr", dtype=dtype)
    v1 = mtspy.matmat(M, v0, use_eigen)
    v2 = M @ v0
    assert(numpy.allclose(v1, v2))


@pytest.mark.parametrize('dtype', dtype_list)
@pytest.mark.parametrize('use_eigen', eigen_backend)
def test_spmv_64bit_ind(dtype, use_eigen):
    N = 1000
    v0 = numpy.random.rand(N, 1).astype(dtype)
    M = sparse.random(N, N, density=0.1, format="csr", dtype=dtype)
    # Change sparse matrix indices dtype
    M.indices = M.indices.astype(numpy.int64)
    M.indptr = M.indptr.astype(numpy.int64)
    v1 = mtspy.matvec(M, v0, use_eigen)
    v2 = M @ v0
    assert(numpy.allclose(v1, v2))


if __name__ == "__main__":
    N = 100
    M = sparse.random(N, N, density=0.1, format="csr")
    m, n = M.shape
    A = mtspy.cpp.sparse_sparse_eigen(m, n, M.nnz, M.data, M.indptr, M.indices,
                                      m, n, M.nnz, M.data, M.indptr, M.indices)

    C = M @ M

    assert((A - C).data.all())
