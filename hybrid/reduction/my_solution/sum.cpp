#include <cstdio>

#define NX 102400

int main(void)
{
    long vecA[NX];
    long sum, psum, sumex;
    int i;

    /* Initialization of the vectors */
    for (i = 0; i < NX; i++) {
        vecA[i] = (long) i + 1;
    }

    sumex = (long) NX * (NX + 1) / ((long) 2);
    printf("Arithmetic sum formula (exact):                  %ld\n",
           sumex);

    sum = 0.0;
    #pragma omp parallel for reduction(+:sum) shared(vecA) private(i)
    for (i = 0; i < NX; i++) {
        sum += vecA[i];
    }
    printf("Sum using reduction: %ld\n", sum);

    sum = 0.0;
    #pragma omp parallel default(shared) private(psum)
    {
        double psum = 0.0;
        #pragma omp for private(i)
        for (i = 0; i < NX; i++) {
            psum += vecA[i];
        }
        #pragma omp critical
        sum += psum;
    }
    printf("Sum using partial sum and critical: %ld\n", sum);

    return 0;
}
