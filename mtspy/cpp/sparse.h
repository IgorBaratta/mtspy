#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pybind11/pybind11.h>
#include <omp.h>

template <typename T, typename I>
Eigen::Matrix<T, Eigen::Dynamic, 1>
SpMV(int rows, int cols, int nnz,
     Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> &data,
     Eigen::Ref<Eigen::Matrix<I, Eigen::Dynamic, 1>> &indptr,
     Eigen::Ref<Eigen::Matrix<I, Eigen::Dynamic, 1>> &indices,
     const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b)
{
    assert(cols == rows);
    Eigen::Matrix<T, Eigen::Dynamic, 1> out(rows);
    if (nnz > 0)
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < rows; ++i)
        {
            const int local_size = indptr[i + 1] - indptr[i];
            auto local_data = data.segment(indptr[i], local_size);
            auto local_cols = indices.segment(indptr[i], local_size);
            out[i] = local_data.transpose() * b(local_cols);
        }

    pybind11::gil_scoped_acquire acquire;
    return out;
}

template <typename T, typename I>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
SpMM(I rows, I cols, I nnz,
     const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> &data,
     const Eigen::Ref<Eigen::Matrix<I, Eigen::Dynamic, 1>> &indptr,
     const Eigen::Ref<Eigen::Matrix<I, Eigen::Dynamic, 1>> &indices,
     Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &dense)
{
    assert(cols == dense.rows());

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> out(cols, dense.cols());
    if (nnz > 0)
    {
#pragma omp parallel for schedule(guided)
        for (I i = 0; i < rows; ++i)
        {
            const int local_size = indptr[i + 1] - indptr[i];
            auto local_data = data.segment(indptr[i], local_size);
            auto local_cols = indices.segment(indptr[i], local_size);
            for (I k = 0; k < dense.cols(); k++)
            {
                auto dense_data = dense(local_cols, k);
                out(i, k) = local_data.transpose() * dense_data;
            }
        }
    }

    pybind11::gil_scoped_acquire acquire;
    return out;
}

template <typename T, typename I>
Eigen::Matrix<T, Eigen::Dynamic, 1>
SPMV_eigen(I rows, I cols, I nnz,
           Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> &data,
           Eigen::Ref<Eigen::Matrix<I, Eigen::Dynamic, 1>> &indptr,
           Eigen::Ref<Eigen::Matrix<I, Eigen::Dynamic, 1>> &indices,
           const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> v1)
{
    // Currently only work with row-major sparse matrix in parallel
    Eigen::Map<const Eigen::SparseMatrix<T, Eigen::RowMajor>> sm1(rows, cols, nnz, indptr.data(), indices.data(), data.data());
    Eigen::Matrix<T, Eigen::Dynamic, 1> v2 = sm1 * v1;
    pybind11::gil_scoped_acquire acquire;
    return v2;
}