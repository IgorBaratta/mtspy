import mtspy_cpp as cpp
import numpy
from scipy import sparse
import warnings


def matvec(A: sparse.spmatrix, x: numpy.ndarray):
    """
    Performs the operation y = A * x where A is an (m, n) sparse matrix
    and x is a column vector or rank-1 array.

    To avoid copies it's recommended that A and x have the same dtype.
    """

    m, n = A.shape

    if sparse.issparse(A):
        if not isinstance(A, sparse.csr_matrix):
            warnings.warn("Converting sparse matrix to CSR",
                          sparse.SparseEfficiencyWarning)
            A = sparse.csr_matrix(A)

    if x.shape != (n,) and x.shape != (n, 1):
        raise ValueError('Dimension mismatch')

    if A.dtype == x.dtype:
        dtype = A.dtype
    elif numpy.iscomplexobj(x) and not numpy.iscomplexobj(A):
        dtype = x.dtype
        A = A.astype(dtype)
        warnings.warn("Converting sparse matrix to complex",
                      sparse.SparseEfficiencyWarning)

    # Convert to row-major (C-style) if not already.
    x = numpy.asanyarray(x, dtype=dtype, order='C')

    if dtype == numpy.float64:
        spmv = cpp.mat_vec_d
    elif dtype == numpy.float32:
        spmv = cpp.mat_vec_f
    elif dtype == numpy.complex64:
        spmv = cpp.mat_vec_cf
    elif dtype == numpy.complex128:
        spmv = cpp.mat_vec_cd
    else:
        raise NotImplementedError

    nnz = A.nnz
    y = spmv(m, n, nnz,
             A.data, A.indptr,
             A.indices, x)

    return y
