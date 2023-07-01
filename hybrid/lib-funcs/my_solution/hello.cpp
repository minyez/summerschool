#include <cstdio>
#include <omp.h>
int main()
{
    int nthreads;
    printf("Hello world!\n");
#pragma omp parallel
    {
        #pragma omp single nowait
        nthreads = omp_get_num_threads();

        printf("Thread: %d\n", omp_get_thread_num());
    }
    printf("Total available threads: %d\n", nthreads);
    return 0;
}
