#pragma once

#include "csr.hpp"
#include "span.hpp"
#include "utils.hpp"

// No floating point operations are performed during the first phase
template <typename ScalarType, typename IndType>
mtspy::csr_data<ScalarType, IndType>
sparse_sparse_two_phases(mtspy::csr_matrix<ScalarType, IndType> &A,
                         mtspy::csr_matrix<ScalarType, IndType> &B,
                         bool compute_data)
{
    // Conservative data allocation
    IndType nrows = A.rows();
    IndType c_nnz = (A.nnz() + B.nnz()) / nrows;

    assert(!compute_data);

    // Alocate intermediate data with rows size
    std::vector<IndType> nnz_per_row(nrows);
    std::vector<std::vector<IndType>> rowmap(nrows);
    std::vector<std::vector<ScalarType>> values(nrows);

    // Initialize vector of "hash tables"
    // Each thread is responsible for one hash table
    mtspy::utils::hash_set<IndType> hash_set;
    hash_set = mtspy::utils::make_table(c_nnz);
    assert((int)hash_set.size() == mtspy::threads::get_max_threads());

    // row by row sparsity pattern computation
#pragma omp parallel for schedule(static) shared(rowmap, hash_set, nnz_per_row)
    for (IndType i = 0; i < nrows; i++)
    {
        // Clear local (thread-owned) hash table
        int id = mtspy::threads::local_id();
        auto &local_hash = hash_set[id];
        local_hash.clear();

        // get local values array
        auto &local_value = values[i];

        // get view to indices of row "i" of matrix A
        mtspy::span<IndType> row_inds = A.row_indices(i);

        for (std::size_t j = 0; j < row_inds.size(); j++)
        {
            // get view to indices of row "J" of matrix B
            const IndType J = row_inds[j];
            mtspy::span<IndType> bJ_inds = B.row_indices(J);
            for (auto &col : bJ_inds)
                auto [it, inserted] = local_hash.insert(col);
        }

        // get nnz of current row and store in nnz_per_row
        nnz_per_row[i] = local_hash.size();

        //insert nonzero indices in rowmap
        rowmap[i].insert(rowmap[i].begin(), local_hash.begin(), local_hash.end());
    }

    // thread-parallel partial sum
    std::vector<IndType> indptr = mtspy::utils::partial_sum(nnz_per_row);

    // flatten rowmap to "indices" of a csr_matrix
    std::vector<IndType> indices = mtspy::utils::flatten_indices(rowmap, indptr);

    // Allocate data for
    std::vector<ScalarType> data(indptr.back(), 0);

    return {data, indptr, indices};
}

//=====================================================================//
// sparse-sparse product using a precomputed pattern and allocated data
template <typename ScalarType, typename IndType>
void sparse_sparse_numeric(mtspy::csr_matrix<ScalarType, IndType> &A,
                           mtspy::csr_matrix<ScalarType, IndType> &B,
                           mtspy::csr_matrix<ScalarType, IndType> &C)
{
    assert(B.cols() == C.cols());
    assert(A.rows() == C.rows());

    // Initialize vector of "hash tables"
    // Each thread is responsible for one hash table
    int num_threads = mtspy::threads::get_max_threads();
    mtspy::utils::hash_map<IndType, IndType> hash_map(num_threads);

    // precompute hashes
    // std::vector<size_t> pre_hashes = mtspy::utils::precompute_hashes(B);
    IndType nrows = C.rows();

#pragma omp parallel for schedule(static) shared(hash_map)
    for (IndType i = 0; i < nrows; i++)
    {
        // Clear local (thread-owned) hash table
        int id = mtspy::threads::local_id();
        auto &local_hash = hash_map[id];
        local_hash.clear();

        // get view to the indices and data of row "i" of matrix C
        mtspy::span<IndType> Ci_inds = C.row_indices(i);
        mtspy::span<ScalarType> Ci_data = C.row_data(i);
        IndType Ci_size = Ci_inds.size();
        local_hash.reserve(Ci_size);

        // Add indices to local hash map, no collisions are exepected
        for (IndType k = 0; k < Ci_size; k++)
            local_hash.insert({Ci_inds[k], k});

        // get view to the indices and data of row "i" of matrix A
        mtspy::span<IndType> Ai_inds = A.row_indices(i);
        mtspy::span<ScalarType> Ai_data = A.row_data(i);
        IndType Ai_size = Ai_inds.size();

        for (IndType k = 0; k < Ai_size; k++)
        {
            IndType K = Ai_inds[k];
            ScalarType value = Ai_data[k];

            // get view to the indices and data of row "K" of matrix B
            mtspy::span<IndType> Bi_inds = B.row_indices(K);
            mtspy::span<ScalarType> Bi_data = B.row_data(K);
            IndType Bi_size = Bi_inds.size();
            for (IndType j = 0; j < Bi_size; j++)
            {
                IndType col = Bi_inds[j];
                IndType cind = local_hash.at(col, col);
                Ci_data[cind] += value * Bi_data[j];
            }
        }
    } // End Parallel region
}