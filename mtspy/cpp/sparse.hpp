#include "csr.hpp"
#include "sparse_sparse_impl.hpp"
#include "utils.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// declares a array_t type of T
template <class T>
using array_t = pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;

template <typename ScalarType, typename IndType>
array_t<ScalarType>
sparse_vec(IndType rows, IndType cols, IndType nnz, const array_t<ScalarType> &data,
           const array_t<IndType> &displ, const array_t<IndType> &indices,
           const array_t<ScalarType> &vec)
{
    // get data pointers
    const ScalarType *data_ptr = data.data();
    const IndType *displ_ptr = displ.data();
    const IndType *indices_ptr = indices.data();
    const ScalarType *vec_ptr = vec.data();

    const IndType vsize = static_cast<IndType>(vec.size());
    if (vsize != cols)
        throw std::runtime_error("Size mismatch.");

    if (nnz != displ_ptr[rows])
        throw std::runtime_error("The sparse matrix data are not consistent.");

    // Allocate output array
    array_t<ScalarType> result(rows);
    auto buffer = result.request(true);
    ScalarType *result_ptr = (ScalarType *)buffer.ptr;

    // Temporarily  release global interpreter lock (GIL)
    pybind11::gil_scoped_release release;

    ScalarType result_i;
#pragma omp parallel for schedule(guided) private(result_i)
    for (IndType i = 0; i < rows; i++)
    {
        const IndType local_size = displ_ptr[i + 1] - displ_ptr[i];
        const ScalarType *current_data = data_ptr + displ_ptr[i];
        const IndType *current_inds = indices_ptr + displ_ptr[i];

        // FIXME: Consider custom reduction(+: result_i) for complex
        result_i = 0;
#pragma omp simd
        for (IndType j = 0; j < local_size; j++)
        {
            const IndType idx = current_inds[j];
            result_i += (current_data[j] * vec_ptr[idx]);
        }

        result_ptr[i] = result_i;
    }

    pybind11::gil_scoped_acquire acquire;

    return result;
}

template <typename ScalarType, typename IndType>
array_t<ScalarType>
sparse_dense(IndType srows, IndType scols, IndType nnz, const array_t<ScalarType> &data,
             const array_t<IndType> &displ, const array_t<IndType> &indices,
             const array_t<ScalarType> &dense)
{
    // get pointers to sparse matrix data
    const ScalarType *data_ptr = data.data();
    const IndType *displ_ptr = displ.data();
    const IndType *indices_ptr = indices.data();

    // get pointers to dense matrix data
    const IndType drows = dense.shape(0);
    const IndType dcols = dense.shape(1);
    const ScalarType *dense_ptr = dense.data();

    // Check data consistency
    if (drows != scols)
        throw std::runtime_error("Size mismatch.");

    if (nnz != displ_ptr[srows])
        throw std::runtime_error("The sparse matrix data are not consistent.");

    // Allocate output array
    array_t<ScalarType> result(srows * dcols);
    auto buffer = result.request(true);
    ScalarType *result_ptr = (ScalarType *)buffer.ptr;
    std::fill(result.mutable_data(), result.mutable_data() + result.size(), 0.);

    // Temporarily  release global interpreter lock (GIL)
    pybind11::gil_scoped_release release;

#pragma omp parallel for schedule(guided)
    for (IndType i = 0; i < srows; i++)
    {
        const IndType local_size = displ_ptr[i + 1] - displ_ptr[i];
        const ScalarType *current_data = data_ptr + displ_ptr[i];
        const IndType *current_inds = indices_ptr + displ_ptr[i];

        for (IndType k = 0; k < local_size; k++)
#pragma omp simd
            for (IndType j = 0; j < dcols; j++)
            {
                const IndType idx = (dcols * current_inds[k]) + j;
                result_ptr[i * dcols + j] += current_data[k] * dense_ptr[idx];
            }
    }

    pybind11::gil_scoped_acquire acquire;

    result.resize({srows, dcols});

    return result;
}

// Gustavson’s algorithm [17]
template <typename ScalarType, typename IndType>
std::tuple<array_t<ScalarType>, array_t<IndType>, array_t<IndType>>
sparse_sparse(IndType m, array_t<ScalarType> &A_data,
              array_t<IndType> &A_displ,
              array_t<IndType> &A_indices,
              array_t<ScalarType> &B_data,
              array_t<IndType> &B_displ,
              array_t<IndType> &B_indices)

{
    // Wrap A-data into a mtspy::csr_matrix
    auto A = mtspy::csr_matrix<ScalarType, IndType>(A_data.mutable_data(), A_displ.mutable_data(),
                                                    A_indices.mutable_data(), {m, m});

    // Wrap B-data into a csr_matrix
    auto B = mtspy::csr_matrix<ScalarType, IndType>(B_data.mutable_data(), B_displ.mutable_data(),
                                                    B_indices.mutable_data(), {m, m});

    // Temporarily  release global interpreter lock (GIL)
    pybind11::gil_scoped_release release;

    // Phase 1 -  symbolic phase - compute sparsity pattern
    auto [indptr, indices] = sparse_sparse_pattern(A, B);
    pybind11::gil_scoped_acquire acquire;

    // Allocate data for output matrix
    array_t<ScalarType> pydata(indptr.back());

    // Wrap B-data into a csr_matrix
    auto C = mtspy::csr_matrix<ScalarType, IndType>(pydata.mutable_data(), indptr.data(),
                                                    indices.data(), {m, m});

    // Phase 2 -  Numerical phase - compute output data for a output
    // matrix C with pre-computed sparsity pattern
    // C = A * B
    sparse_sparse_product<ScalarType, IndType>(A, B, C);

    // TODO: create function to pefrome this operation
    // move indptr to numpy array
    auto v = new std::vector<IndType>(indptr);
    auto capsule = pybind11::capsule(v, [](void *v) { delete reinterpret_cast<std::vector<IndType> *>(v); });
    auto pyindptr = pybind11::array(v->size(), v->data(), capsule);

    // move indidces to numpy array
    auto v1 = new std::vector<IndType>(indices);
    auto capsule1 = pybind11::capsule(v1, [](void *v1) { delete reinterpret_cast<std::vector<IndType> *>(v1); });
    auto pyindices = pybind11::array(v1->size(), v1->data(), capsule1);

    return {pydata, pyindptr, pyindices};
}