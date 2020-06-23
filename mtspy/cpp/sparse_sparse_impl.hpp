#pragma once

#include "csr.hpp"
#include "span.hpp"
#include "utils.hpp"
#include <iostream>

// No floating point operations are performed during the first phase
template <typename ScalarType, typename IndType>
mtspy::sparsity_pattern<IndType>
sparse_sparse_pattern(mtspy::csr_matrix<ScalarType, IndType> &A,
                      mtspy::csr_matrix<ScalarType, IndType> &B)
{
    // Initialize output sparsitpy pattern struct
    mtspy::sparsity_pattern<IndType> pattern;

    // Conservative data allocation
    IndType nrows = A.rows();
    IndType c_nnz = (A.nnz() + B.nnz()) / nrows;

    // Alocate intermediate data with rows size
    std::vector<IndType> nnz_per_row(A.rows());
    std::vector<std::vector<IndType>> rowmap(nrows);

    // Initialize vector of "hash tables"
    // Each thread is responsible for one hash table
    mtspy::utils::hash_table<IndType> hash_table;
    hash_table = mtspy::utils::make_table(c_nnz);
    assert((int)hash_table.size() == mtspy::threads::get_max_threads());

    // row by row sparsity pattern computation
#pragma omp parallel for schedule(static) shared(rowmap, hash_table, nnz_per_row)
    for (IndType i = 0; i < nrows; i++)
    {
        // Clear local (thread-owned) hash table
        int id = mtspy::threads::local_id();
        auto &local_hash = hash_table[id];
        local_hash.clear();

        // get view to indices of row "i" of matrix A
        mtspy::span<IndType> row_inds = A.row_indices(i);

        for (std::size_t j = 0; j < row_inds.size(); j++)
        {
            // get view to indices of row "J" of matrix B
            const IndType J = row_inds[j];
            mtspy::span<IndType> bJ_inds = B.row_indices(J);
            local_hash.insert(bJ_inds.begin(), bJ_inds.end());
        }

        // get nnz of current row and store in nnz_per_row
        IndType nnz_row = local_hash.size();
        nnz_per_row[i] = nnz_row;

        //insert nonzero indices in rowmap
        rowmap[i].insert(rowmap[i].begin(), local_hash.begin(), local_hash.end());
    }

    // thread-parallel partial sum
    std::vector<IndType> indptr = mtspy::utils::partial_sum(nnz_per_row);

    // flatten rowmap to "indices" of a csr_matrix
    std::vector<IndType> indices = mtspy::utils::flatten_indices(rowmap, indptr);

    return {indptr, indices};
}

template <typename ScalarType, typename IndType>
void sparse_sparse_product(mtspy::csr_matrix<ScalarType, IndType> &A,
                           mtspy::csr_matrix<ScalarType, IndType> &B,
                           mtspy::csr_matrix<ScalarType, IndType> &C)
{

    IndType nrows = C.rows();
    assert(B.cols() == C.cols());
    robin_hood::unordered_flat_set<IndType> hash_table;

    for (IndType i = 0; i < nrows; i++)
    {
        hash_table.clear();
        mtspy::span<IndType> Ai_inds = A.row_indices(i);
        mtspy::span<IndType> Ci_inds = C.row_indices(i);
        mtspy::span<ScalarType> Ci = C.row_data(i);
        auto dx = Ai_inds.size();
        auto dy = Ci_inds.size();
        // ScalarType acc[dx][dy];
        // hash_table.rehash(Ci_inds.size());
        // hash_table.insert(Ci_inds.begin(), Ci_inds.end());
        // for (size_t k = 0; k < dx; k++)
        // {
        //     // auto c = hash_table.find(Ai_inds[k]);
        //     // auto pos = std::distance(hash_table.begin(), c);
        //     // std::cout << pos << std::endl;
        //     acc[0][k] = k;
        // }
    }
}