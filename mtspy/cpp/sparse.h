#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pybind11/pybind11.h>
#include <omp.h>

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
mat_vec(int rows, int cols, int nnz,
        Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> &data,
        Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, 1>> &indptr,
        Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, 1>> &indices,
        const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> v1)
{
    // Currently only work with row-major sparse matrix in parallel
    Eigen::Map<const Eigen::SparseMatrix<T, Eigen::RowMajor>> sm1(rows, cols, nnz, indptr.data(), indices.data(), data.data());
    Eigen::Matrix<T, Eigen::Dynamic, 1> v2 = sm1 * v1;
    pybind11::gil_scoped_acquire acquire;
    return v2;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
mat_vec_u(const int local_size, int blocks,
          Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> &data,
          Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, 1>> &indices,
          const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> v1)
{
    if (local_size < 2 or blocks < 2)
        throw std::runtime_error("message");

    Eigen::Matrix<T, Eigen::Dynamic, 1> ext_vec(blocks * local_size, 0);
#pragma omp parallel for
    for (int i = 0; i < blocks; i++)
    {
        auto local_data = data.segment(i, local_size * local_size);
        auto local_vec = ext_vec.segment(i, local_size);
        auto local_indice = indices.segment(i, local_size);
        for (int j = 0; j < local_size; j++)
            for (int k = 0; k < local_size; k++)
                ext_vec[j] += local_data[j * local_size + k] * v1[local_indice[j * local_size + k]];
    }
}