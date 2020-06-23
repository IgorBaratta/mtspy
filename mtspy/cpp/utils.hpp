#pragma once

#include "thread_control.hpp"
#include "tsl/robin_growth_policy.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include <functional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <type_traits>
#include <vector>

namespace mtspy::utils
{

    //====================================================================================//
    template <typename IndType>
    std::vector<IndType> partial_sum(std::vector<IndType> &input)
    {
        IndType isize = input.size();
        std::vector<IndType> output(isize + 1);

        output[0] = 0;

        for (IndType i = 0; i < isize; i++)
        {
            output[i + 1] = output[i] + input[i];
        }
        return output;
    }

    //====================================================================================//
    template <typename Key>
    using hash_set = std::vector<tsl::robin_set<Key, std::hash<Key>,
                                                std::equal_to<Key>,
                                                std::allocator<Key>,
                                                false>>;

    template <typename Key, typename T>
    // using hash_map = std::vector<tsl::hopscotch_map<Key, T>>;
    // using hash_map = std::vector<tsl::robin_map<Key, T>>;
    using hash_map = std::vector<tsl::robin_map<Key, T,
                                                std::hash<Key>,
                                                std::equal_to<Key>,
                                                std::allocator<std::pair<Key, T>>,
                                                true>>;

    //====================================================================================//
    template <typename IndType>
    hash_set<IndType> make_table(IndType conservative_nnz)
    {
        int nthreads = mtspy::threads::get_max_threads();
        hash_set<IndType> table(nthreads);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < nthreads; i++)
        {
            table[i].rehash(conservative_nnz);
            table[i].reserve(conservative_nnz);
        }

        return table;
    }

    //====================================================================================//
    template <typename IndType>
    std::vector<IndType> flatten_indices(std::vector<std::vector<IndType>> &row_map,
                                         std::vector<IndType> &indptr)
    {
        IndType nrows = indptr.size() - 1;
        IndType nnz = indptr.back();
        std::vector<IndType> flattened_output(nnz);

#pragma omp parallel for schedule(guided)
        for (IndType i = 0; i < nrows; i++)
        {
            std::vector<IndType> &rowmap_i = row_map[i];
            IndType local_size = indptr[i + 1] - indptr[i];
            IndType pos_i = indptr[i];
#pragma omp simd
            for (IndType j = 0; j < local_size; j++)
            {
                flattened_output[pos_i + j] = rowmap_i[j];
            }
        }

        return flattened_output;
    }
    //====================================================================================//

    /// Precompute hashes of stored indices of columns B
    template <typename ScalarType, typename IndType>
    std::vector<size_t> precompute_hashes(mtspy::csr_matrix<ScalarType, IndType> &B)
    {
        // precompute hashes of columns of B
        std::vector<size_t> precomputed_hashes(B.nnz());
        const IndType *indices = B.indices();

#pragma omp for schedule(static)
        for (IndType i = 0; i < B.nnz(); i++)
            precomputed_hashes[i] = std::hash<IndType>()(indices[i]);

        return precomputed_hashes;
    }

} // namespace mtspy::utils