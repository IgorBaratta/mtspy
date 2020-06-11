#include <assert.h>
#include <omp.h>

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