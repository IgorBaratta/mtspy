#pragma once

#include "span.hpp"
#include <array>

namespace mtspy
{
    // CSR matrix wrapper using the standard representation where the column indices for
    // row i are stored in indices and their values are stored in data.
    template <typename ScalarType, typename IndType>
    class csr_matrix
    {
        ScalarType *_data;
        IndType *_indptr;
        IndType *_indices;
        std::array<IndType, 2> _shape;

    public:
        /// Constructs a CSR matrix wrapper from existing data
        csr_matrix(ScalarType *data, IndType *indptr,
                   IndType *indices, std::array<IndType, 2> shape) noexcept
            : _data{data}, _indptr{indptr}, _indices{indices}, _shape{shape} {}

        /// Accesses an element of the sequence, do not check bounds
        ScalarType &operator[](int i) noexcept { return _data[i]; }

        /// Return the number of rows
        IndType rows() const noexcept { return _shape[0]; }

        /// Return the number of columns
        IndType cols() const noexcept { return _shape[1]; }

        /// Returns the number of stored nonzeros
        IndType nnz() const noexcept { return _indptr[_shape[0]]; }

        /// Returns a pointer to the beginning of the data
        const ScalarType *data() noexcept { return _data; }

        /// Returns a pointer to the beginning of the indptr
        const IndType *indptr() noexcept { return _indptr; }

        /// Returns a pointer to the beginning of the indices
        const IndType *indices() noexcept { return _indices; }

        /// Return view to the data of row i
        span<ScalarType> row_data(IndType i) noexcept
        {
            assert(i < _shape[0]);
            ScalarType *row_data = _data + _indptr[i];
            IndType row_size = _indptr[i + 1] - _indptr[i];
            span<ScalarType> data_view(row_data, row_size);
            return data_view;
        }

        /// Return view to indices (cols) of stored elements in row i
        span<IndType> row_indices(IndType i) noexcept
        {
            assert(i < _shape[0]);
            IndType *row_inds = _indices + _indptr[i];
            IndType row_size = _indptr[i + 1] - _indptr[i];
            span<IndType> ind_view(row_inds, row_size);
            return ind_view;
        }
    };

    /// Simple csr data sctructure used for structured binding in C++ 17
    template <typename ScalarType, typename IndType>
    struct csr_data
    {
        std::vector<ScalarType> data;
        std::vector<IndType> indptr;
        std::vector<IndType> indices;
    };

} // namespace mtspy