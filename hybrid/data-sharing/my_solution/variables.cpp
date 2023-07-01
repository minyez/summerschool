#include <cstdio>
#include <omp.h>

int main(void)
{
    int var1 = 1, var2 = 2;

    // #pragma omp parallel shared(var1,var2)
    // #pragma omp parallel private(var1,var2)
    #pragma omp parallel lastprivate(var1,var2)
    {
        printf("Thread %2d Region 1: var1=%i, var2=%i\n", omp_get_thread_num(), var1, var2);
        var1++;
        var2++;
    }
    printf("After region 1: var1=%i, var2=%i\n\n", var1, var2);

    return 0;
}
