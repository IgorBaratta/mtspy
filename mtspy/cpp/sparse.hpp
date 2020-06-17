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

#pragma omp parallel for schedule(guided)
    for (IndType i = 0; i < rows; i++)
    {
        const IndType local_size = displ_ptr[i + 1] - displ_ptr[i];
        const ScalarType *current_data = data_ptr + displ_ptr[i];
        const IndType *current_inds = indices_ptr + displ_ptr[i];

        // FIXME: Provide custom reduction(+: result_i) for complex
        ScalarType result_i = 0;
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