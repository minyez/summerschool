#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
    if (err != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// TODO: add a device kernel that calculates y = a * x + y
__global__ void saxpy(float *y, const float *x, float a, int n)
{
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < n)
        y[ind] = y[ind] + a * x[ind];
}

int main(void)
{
    int i;
    const int n = 10000;
    float a = 3.4;
    float x[n], y[n], y_ref[n];
    float *x_, *y_;

    // initialise data and calculate reference values on CPU
    for (i=0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
        y_ref[i] = a * x[i] + y[i];
    }

    // TODO: allocate vectors x_ and y_ on the GPU
    // TODO: copy initial values from CPU to GPU (x -> x_ and y -> y_)
    HIP_ERRCHK(hipMalloc(&x_, n * sizeof(float)));
    HIP_ERRCHK(hipMalloc(&y_, n * sizeof(float)));
    HIP_ERRCHK(hipMemcpy(x_, x, n * sizeof(float), hipMemcpyHostToDevice));
    HIP_ERRCHK(hipMemcpy(y_, y, n * sizeof(float), hipMemcpyHostToDevice));

    dim3 threads(256);
    dim3 blocks(64);
    // int threads = 128;
    // int blocks = (n + threads - 1) / threads;
    // blocks * threads >= n, otherwise large error
    hipLaunchKernelGGL(saxpy, blocks, threads, 0, 0, y_, x_, a, n);
    // saxpy<<<blocks, threads>>>(y, x, a, n);

    // TODO: copy results back to CPU (y_ -> y)
    HIP_ERRCHK(hipDeviceSynchronize());
    HIP_ERRCHK(hipMemcpy(y, y_, n * sizeof(float), hipMemcpyDeviceToHost));

    HIP_ERRCHK(hipFree(x_));
    HIP_ERRCHK(hipFree(y_));

    // confirm that results are correct
    float error = 0.0;
    float tolerance = 1e-6;
    float diff;
    for (i=0; i < n; i++) {
        diff = abs(y_ref[i] - y[i]);
        if (diff > tolerance)
            error += diff;
    }
    printf("total error: %f\n", error);
    printf("  reference: %f at (42)\n", y_ref[42]);
    printf("     result: %f at (42)\n", y[42]);

    return 0;
}
