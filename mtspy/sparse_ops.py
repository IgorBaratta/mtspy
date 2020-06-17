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

    spmv = cpp.sparse_vec

    # If use_eigen is true try to use as a computational backend,
    # fallback to built-in spmv if it's not linked
    if use_eigen:
        try:
            spmv = cpp.sparse_dense_eigen
        except AttributeError:
            print("Eigen not available, using built-in backend.")

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

    spmm = cpp.sparse_dense
    # If use_eigen is true try to use as a computational backend,
    # fallback to built-in spmv if it's not linked
    if use_eigen:
        try:
            spmm = cpp.sparse_dense_eigen
        except AttributeError:
            print("Eigen not available, using built-in backend instead.")

    Y = spmm(m, n, A.nnz, A.data, A.indptr, A.indices, X)

    return Y


def spmatmat(A: sparse.spmatrix, B: sparse.spmatrix , use_eigen: bool = True) -> numpy.ndarray:
    """
    Performs the operation C = A * B,  where A is a (m, k) sparse matrix
    and B is a (k, n) sparse matrix.
    """

    if use_eigen:
        try:
            spmspm = cpp.sparse_sparse_eigen
        except AttributeError:
            print("Eigen not available.")
    else:
        raise NotImplementedError

    C = spmspm(A.shape[0], A.shape[1], A.nnz, A.data, A.indptr, A.indices,
               B.shape[0], B.shape[1], B.nnz, B.data, B.indptr, B.indices,)

    return C
