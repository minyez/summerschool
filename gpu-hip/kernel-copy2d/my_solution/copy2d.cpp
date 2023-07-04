#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
    if (err != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// TODO: add a device kernel that copies all elements of a vector
//       using GPU threads in a 2D grid
__global__ void copy2d(double* dst, double* src, int n, int m)
{
    int indx = threadIdx.x + blockIdx.x * blockDim.x;
    int indy = threadIdx.y + blockIdx.y * blockDim.y;
    int ind = indy * n + indx;
    const int size = n * m;
    if (ind < size)
        dst[ind] = src[ind];
}


int main(void)
{
    int i, j;
    const int n = 600;
    const int m = 400;
    const int size = n * m;
    double x[size], y[size], y_ref[size];
    double *x_, *y_;

    // initialise data
    for (i=0; i < size; i++) {
        x[i] = (double) i / 1000.0;
        y[i] = 0.0;
    }
    // copy reference values (C ordered)
    for (i=0; i < n; i++) {
        for (j=0; j < m; j++) {
            y_ref[i * m + j] = x[i * m + j];
        }
    }

    // TODO: allocate vectors x_ and y_ on the GPU
    HIP_ERRCHK(hipMalloc(&x_, n * m * sizeof(double)));
    HIP_ERRCHK(hipMalloc(&y_, n * m * sizeof(double)));
    // TODO: copy initial values from CPU to GPU (x -> x_ and y -> y_)
    HIP_ERRCHK(hipMemcpy(x_, x, n * m * sizeof(double), hipMemcpyHostToDevice));
    HIP_ERRCHK(hipMemcpy(y_, y, n * m * sizeof(double), hipMemcpyHostToDevice));

    // TODO: define grid dimensions (use 2D grid!)
    dim3 threads(16, 16, 1);
    dim3 blocks(64, 64, 1);
    // TODO: launch the device kernel
    hipLaunchKernelGGL(copy2d, blocks, threads, 0, 0, y_, x_, n, m);

    // TODO: copy results back to CPU (y_ -> y)
    // HIP_ERRCHK(hipDeviceSynchronize());
    HIP_ERRCHK(hipMemcpy(y, y_, n * m * sizeof(double), hipMemcpyDeviceToHost));

    // confirm that results are correct
    double error = 0.0;
    for (i=0; i < size; i++) {
        error += abs(y_ref[i] - y[i]);
    }
    printf("total error: %f\n", error);
    printf("  reference: %f at (42,42)\n", y_ref[42 * m + 42]);
    printf("     result: %f at (42,42)\n", y[42 * m + 42]);

    return 0;
}
