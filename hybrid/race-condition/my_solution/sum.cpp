#include <cstdio>
#include <omp.h>

#define NX 102400

int main(void)
{
    long vecA[NX];
    long sum, psum, sumex;
    int i;

    double tstart;

    /* Initialization of the vectors */
    for (i = 0; i < NX; i++) {
        vecA[i] = (long) i + 1;
    }

    sumex = (long) NX * (NX + 1) / ((long) 2);
    printf("Arithmetic sum formula (exact):                  %ld\n",
           sumex);

    sum = 0.0;
    tstart = omp_get_wtime();
    // #pragma omp parallel for reduction(+:sum) shared(vecA) private(i)
    #pragma omp parallel for shared(vecA) private(i)
    for (i = 0; i < NX; i++) {
        sum += vecA[i];
    }
    printf("Sum: %ld\n", sum);
    printf("Wall time: %f\n", omp_get_wtime() - tstart);

    return 0;
}
