#pragma once

#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#define omp_set_num_threads(n) (0)
#endif

// nesting is not supported
namespace mtspy::threads
{

    int get_max_threads()
    {
        return omp_get_max_threads();
    }

    int get_num_threads()
    {
        return omp_get_num_threads();
    }

    int local_id()
    {
        return omp_get_thread_num();
    }

    void set_num_threads(int n)
    {
        assert(n > 0);
        omp_set_num_threads(n);
    }

} // namespace mtspy::threads