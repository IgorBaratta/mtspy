#pragma once

#include "robin_hood.h"
#include "thread_control.hpp"
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
    template <typename T>
    using hash_table = std::vector<robin_hood::unordered_set<T>>;

    //====================================================================================//
    template <typename IndType>
    hash_table<IndType> make_table(IndType conservative_nnz)
    {
        int nthreads = mtspy::threads::get_max_threads();
        hash_table<IndType> table(nthreads);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < nthreads; i++)
            table[i].rehash(conservative_nnz);

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

} // namespace mtspy::utils