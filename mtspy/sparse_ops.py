import mtspy_cpp as cpp
import numpy
from scipy import sparse
import warnings


def matvec(A: sparse.spmatrix, x: numpy.ndarray, use_eigen=False) -> numpy.ndarray:
    """
    Performs the operation y = A * x where A is an (m, n) sparse matrix
    and x is a column vector or rank-1 array of size m.

    To avoid copies it's recommended that A and x have the same dtype.
    """

    m, n = A.shape

    if sparse.issparse(A):
        if not isinstance(A, sparse.csr_matrix):
            warnings.warn("Converting sparse matrix to CSR",
                          sparse.SparseEfficiencyWarning)
            A = sparse.csr_matrix(A)

    if x.shape == (n,):
        output_shape = (m,)
    elif x.shape == (n, 1):
        output_shape = (m, 1)
    else:
        raise ValueError('Dimension mismatch')

    if numpy.iscomplexobj(x) and not numpy.iscomplexobj(A):
        A = A.astype(x.dtype)
        warnings.warn("Converting sparse matrix to complex",
                      sparse.SparseEfficiencyWarning)

    # Convert to row-major (C-style) if it's not already.
    x = numpy.asanyarray(x, dtype=A.dtype, order='C')

    if use_eigen:
        # Use Eigen backend
        spmv = cpp.spmm_eigen
    else:
        spmv = cpp.spmv

    y = spmv(m, n, A.nnz,
             A.data, A.indptr,
             A.indices, x)

    return numpy.reshape(y, output_shape, 'C')


def matmat(A: sparse.spmatrix, X: numpy.ndarray, use_eigen: bool = False) -> numpy.ndarray:
    """
    Performs the operation C = A * B,  where A is a (m, k) sparse matrix
    and B is a (k, n) dense matrix.


    To avoid copies it's recommended that A and X have the same dtype.
    """

    m, n = A.shape

    if sparse.issparse(A):
        if not isinstance(A, sparse.csr_matrix):
            warnings.warn("Converting sparse matrix to CSR",
                          sparse.SparseEfficiencyWarning)
            A = sparse.csr_matrix(A)

    if X.shape[0] != n:
        raise ValueError('Dimension mismatch')

    if A.dtype == X.dtype:
        dtype = A.dtype
    elif numpy.iscomplexobj(X) and not numpy.iscomplexobj(A):
        dtype = X.dtype
        A = A.astype(dtype)
        warnings.warn("Converting sparse matrix to complex",
                      sparse.SparseEfficiencyWarning)

    # Convert to row-major (C-style) if it's not already.
    X = numpy.asanyarray(X, dtype=dtype, order='C')

    if use_eigen:
        # Use Eigen backend
        # TODO: Remove Eigen backend
        spmm = cpp.spmm_eigen
    else:
        spmm = cpp.spmm

    Y = spmm(m, n, A.nnz,
             A.data, A.indptr,
             A.indices, X)

    return Y
