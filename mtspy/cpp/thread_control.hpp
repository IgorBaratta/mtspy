#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_set_num_threads(n) (0)
#endif

int get_max_threads()
{
    return omp_get_max_threads();
}

int get_num_threads()
{
    return omp_get_num_threads();
}

void set_num_threads(int n)
{
    assert(n > 0);
    omp_set_num_threads(n);
}