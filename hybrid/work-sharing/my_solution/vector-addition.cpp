#include <cstdio>
#include <ctime>
#include <omp.h>

#define NX 204800

int main(void)
{
    double vecA[NX], vecB[NX], vecC[NX];
    double t, tst;
    clock_t ct1, ct2;

    /* Initialization of the vectors */
    for (int i = 0; i < NX; i++) {
        vecA[i] = 1.0 / ((double)(NX - i));
        vecB[i] = vecA[i] * vecA[i];
    }

#pragma omp parallel
    {
    printf("Thread ID: %d\n", omp_get_thread_num());
    }

    /* DONE:
     *   Implement here a parallelized version of vector addition,
     *   vecC = vecA + vecB
     */
    ct1 = clock();
    tst = omp_get_wtime();

// #pragma omp parallel for
    #pragma omp parallel for default(shared)
    for (int i = 0; i < NX; i++) {
        vecC[i] = vecA[i] + vecB[i];
    }

    ct2 = clock();
    t = double(ct2 - ct1) / CLOCKS_PER_SEC;

    printf("array+array CPU  time: %f\n", t);
    printf("array+array Wall time: %f\n", omp_get_wtime() - tst);

    double sum = 0.0;
    /* Compute the check value */
    // ct1 = clock();
    // tst = omp_get_wtime();
    for (int i = 0; i < NX; i++) {
        sum += vecC[i];
    }
    // ct2 = clock();
    printf("Reduction sum: %18.16f\n", sum);
    // t = double(ct2 - ct1) / CLOCKS_PER_SEC;
    // printf("Array-scalar CPU  time: %f\n", t);
    // printf("Array-scalar Wall time: %f\n", omp_get_wtime() - tst);

    return 0;
}
