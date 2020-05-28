#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pybind11/pybind11.h>
#include <omp.h>


Eigen::VectorXd mat_vec(int rows, int cols, int nnz,
                        Eigen::Ref<Eigen::VectorXd> &data,
                        Eigen::Ref<Eigen::VectorXi> &indptr,
                        Eigen::Ref<Eigen::VectorXi> &indices,
                        const Eigen::Ref<Eigen::VectorXd> v1)
{
    // Currently only work with row-major sparse matrix in parallel
    Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>> sm1(rows, cols, nnz, indptr.data(), indices.data(), data.data());
    Eigen::VectorXd v2(v1.size());
    v2 = sm1 * v1;
    pybind11::gil_scoped_acquire acquire;
    return v2;
}
