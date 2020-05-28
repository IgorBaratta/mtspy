#include <omp.h>
#include <Eigen/Core>

int max_threads()
{
    return omp_get_max_threads();
}
