import numpy
from scipy import sparse
import pytest
import warnings

import mtspy

dtype_list = [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]
if mtspy.cpp.has_eigen():
    eigen_backend = [False, True]
else:
    eigen_backend = [False]


@pytest.mark.parametrize('dtype', dtype_list)
@pytest.mark.parametrize('use_eigen', eigen_backend)
def test_sparse_vec_int32(dtype, use_eigen):
    N = 1000
    v0 = numpy.random.rand(N, 1).astype(dtype)
    M = sparse.random(N, N, density=0.1, format="csr", dtype=dtype)
    v1 = mtspy.matvec(M, v0, use_eigen)
    v2 = M @ v0
    assert(numpy.allclose(v1, v2))


@pytest.mark.parametrize('dtype', dtype_list)
@pytest.mark.parametrize('use_eigen', eigen_backend)
def test_sparse_vec_int64(dtype, use_eigen):
    N = 1000
    v0 = numpy.random.rand(N, 1).astype(dtype)
    M = sparse.random(N, N, density=0.1, format="csr", dtype=dtype)
    # Change sparse matrix indices dtype
    M.indices = M.indices.astype(numpy.int64)
    M.indptr = M.indptr.astype(numpy.int64)
    v1 = mtspy.matvec(M, v0, use_eigen)
    v2 = M @ v0
    assert(numpy.allclose(v1, v2))


@pytest.mark.parametrize('dtype', dtype_list)
@pytest.mark.parametrize('use_eigen', eigen_backend)
def test_sparse_dense_int32(dtype, use_eigen):
    N = 1000
    v0 = numpy.ones((N, 10), dtype=dtype)
    M = sparse.random(N, N, density=0.1, format="csr", dtype=dtype)
    v1 = mtspy.matmat(M, v0, use_eigen)
    v2 = M @ v0
    assert(numpy.allclose(v1, v2))


@pytest.mark.parametrize('dtype', dtype_list)
@pytest.mark.skipif(not mtspy.cpp.has_eigen(), reason="Sparse-Sparse Matrix Product only supported with eigen backend.")
def test_sparse_sparse_int32(dtype):
    m, n, k = 500, 1000, 2000
    A = sparse.random(m, k, density=0.1, format="csr")
    B = sparse.random(k, n, density=0.1, format="csr")

    C1 = mtspy.spmatmat(A, B, True)
    C2 = A @ B
    assert((C1 - C2).data.all())


@pytest.mark.parametrize('dtype', dtype_list)
@pytest.mark.parametrize('use_eigen', eigen_backend)
def test_sparse_csc_vec_int32(recwarn, dtype, use_eigen):
    warnings.simplefilter("always")
    N = 100
    v0 = numpy.random.rand(N, 1).astype(dtype)
    M = sparse.random(N, N, density=0.1, format="csc", dtype=dtype)
    v1 = mtspy.matvec(M, v0, use_eigen)
    v2 = M @ v0
    assert(numpy.allclose(v1, v2))
    assert len(recwarn) == 1
    assert recwarn.pop(sparse.SparseEfficiencyWarning)


def test_sparse_ops_errors():
    v0 = numpy.random.rand(10, 1)
    M = sparse.random(5, 5, density=0.1, format="csr")
    with pytest.raises(ValueError):
        v1 = mtspy.matvec(M, v0)
        v1 = mtspy.matmat(M, v0)


def test_complex_warning(recwarn):
    M = sparse.random(5, 5, density=0.1, format="csr")
    v0 = numpy.random.rand(M.shape[0], 1)
    v0 = v0.astype(numpy.complex128)
    v1 = mtspy.matvec(M, v0)
    v2 = mtspy.matmat(M, v0)
    assert len(recwarn) == 2
    assert recwarn.pop(sparse.SparseEfficiencyWarning)
