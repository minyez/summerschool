#include <cstdio>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>

#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
    if (err != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// Data structure for storing decomposition information
struct Decomp {
    int len;    // length of the array for the current device
    int start;  // start index for the array on the current device
};


/* HIP kernel for the addition of two vectors, i.e. C = A + B */
__global__ void vector_add(double *C, const double *A, const double *B, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Do not try to access past the allocated memory
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}


int main(int argc, char *argv[])
{
    const int ThreadsInBlock = 128;
    double *dA[2], *dB[2], *dC[2];
    double *hA, *hB, *hC;
    int devicecount;
    int N = 100;
    int GridSize;
    hipEvent_t start, stop;
    hipEvent_t start_h2d_event[2], start_kernel_event[2], start_d2h_event[2], start_sync_event[2];
    hipEvent_t stop_h2d_event[2], stop_kernel_event[2], stop_d2h_event[2], stop_sync_event[2];
    hipStream_t strm[2];
    Decomp dec[2];

    // Check that we have two HIP devices available
    HIP_ERRCHK(hipGetDeviceCount(&devicecount));
    if (devicecount != 2)
    {
        printf("require 2 GPUs, have %d.\n", devicecount);
        return EXIT_FAILURE;
    }

    // Create timing events
    HIP_ERRCHK(hipSetDevice(0));
    HIP_ERRCHK(hipEventCreate(&start));
    HIP_ERRCHK(hipEventCreate(&stop));
    for (int i = 0; i < 2; ++i)
    {
        HIP_ERRCHK(hipSetDevice(i));
        HIP_ERRCHK(hipEventCreate(start_h2d_event+i));
        HIP_ERRCHK(hipEventCreate(start_kernel_event+i));
        HIP_ERRCHK(hipEventCreate(start_d2h_event+i));
        HIP_ERRCHK(hipEventCreate(start_sync_event+i));
        HIP_ERRCHK(hipEventCreate(stop_h2d_event+i));
        HIP_ERRCHK(hipEventCreate(stop_kernel_event+i));
        HIP_ERRCHK(hipEventCreate(stop_d2h_event+i));
        HIP_ERRCHK(hipEventCreate(stop_sync_event+i));
    }

    HIP_ERRCHK(hipSetDevice(0));
    // Allocate host memory
    // Allocate enough pinned host memory for hA, hB, and hC
    // to store N doubles each
    int nbytes = N * sizeof(double);
    HIP_ERRCHK(hipHostMalloc(&hA, nbytes));
    HIP_ERRCHK(hipHostMalloc(&hB, nbytes));
    HIP_ERRCHK(hipHostMalloc(&hC, nbytes));

    // Initialize host memory
    for(int i = 0; i < N; ++i) {
        hA[i] = 1.0;
        hB[i] = 2.0;
    }

    // Decomposition of data for each stream
    dec[0].len   = N / 2;
    dec[0].start = 0;
    dec[1].len   = N - N / 2;
    dec[1].start = dec[0].len;

    // Allocate memory for the devices and per device streams
    for (int i = 0; i < 2; ++i) {
        hipSetDevice(i);
        // Allocate enough device memory for dA[i], dB[i], dC[i]
        // to store dec[i].len doubles
        HIP_ERRCHK(hipMalloc(dA+i, dec[i].len * sizeof(double)));
        HIP_ERRCHK(hipMalloc(dB+i, dec[i].len * sizeof(double)));
        HIP_ERRCHK(hipMalloc(dC+i, dec[i].len * sizeof(double)));
        // Create a stream for each device
        HIP_ERRCHK(hipStreamCreate(strm+i));
    }

    // Start timing
    // auto start_clock = std::chrono::steady_clock::now();
    auto start_clock = std::chrono::system_clock::now();
    HIP_ERRCHK(hipSetDevice(0));
    HIP_ERRCHK(hipEventRecord(start));

    /* Copy each decomposed part of the vectors from host to device memory
       and execute a kernel for each part.
       Note: one needs to use streams and asynchronous calls! Without this
       the execution is serialized because the memory copies block the
       execution of the host process. */
    for (int i = 0; i < 2; ++i) {
        // Set active device
        HIP_ERRCHK(hipSetDevice(i));
        HIP_ERRCHK(hipEventRecord(start_h2d_event[i], strm[i]));
        // Copy data from host to device asynchronously (hA[dec[i].start] -> dA[i], hB[dec[i].start] -> dB[i])
        HIP_ERRCHK(hipMemcpyAsync(dA[i], hA + dec[i].start, dec[i].len * sizeof(double), hipMemcpyHostToDevice, strm[i]));
        HIP_ERRCHK(hipMemcpyAsync(dB[i], hB + dec[i].start, dec[i].len * sizeof(double), hipMemcpyHostToDevice, strm[i]));
        HIP_ERRCHK(hipEventRecord(stop_h2d_event[i], strm[i]));
        // Launch 'vector_add()' kernel to calculate dC = dA + dB
        GridSize = (dec[i].len + ThreadsInBlock - 1) / ThreadsInBlock;
        HIP_ERRCHK(hipEventRecord(start_kernel_event[i], strm[i]));
        hipLaunchKernelGGL(vector_add, GridSize, ThreadsInBlock, 0, strm[i], dC[i], dA[i], dB[i], dec[i].len);
        HIP_ERRCHK(hipEventRecord(stop_kernel_event[i], strm[i]));
        // Copy data from device to host (dC[i] -> hC[dec[0].start])
        HIP_ERRCHK(hipEventRecord(start_d2h_event[i], strm[i]));
        HIP_ERRCHK(hipMemcpyAsync(hC + dec[i].start, dC[i], dec[i].len * sizeof(double), hipMemcpyDeviceToHost, strm[i]));
        HIP_ERRCHK(hipEventRecord(stop_d2h_event[i], strm[i]));
    }

    // Synchronize and destroy the streams
    for (int i = 0; i < 2; ++i) {
        HIP_ERRCHK(hipSetDevice(i));
        // Add synchronization calls and destroy streams
        HIP_ERRCHK(hipEventRecord(start_sync_event[i], strm[i]));
        HIP_ERRCHK(hipStreamSynchronize(strm[i]));
        HIP_ERRCHK(hipEventRecord(stop_sync_event[i], strm[i]));
        HIP_ERRCHK(hipStreamDestroy(strm[i]));
    }

    // Stop timing
    // Add here the timing event stop calls
    HIP_ERRCHK(hipSetDevice(0));
    HIP_ERRCHK(hipEventRecord(stop));
    // auto stop_clock = std::chrono::steady_clock::now();
    auto stop_clock = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = stop_clock - start_clock;
    printf("Time elapsed by chrono: %f\n", duration.count());

    // Free device memory
    for (int i = 0; i < 2; ++i) {
        // Deallocate device memory
        HIP_ERRCHK(hipSetDevice(i));
        HIP_ERRCHK(hipFree(dA[i]));
        HIP_ERRCHK(hipFree(dB[i]));
        HIP_ERRCHK(hipFree(dC[i]));
    }

    // Check results
    int errorsum = 0;
    for (int i = 0; i < N; i++) {
        errorsum += hC[i] - 3.0;
    }
    printf("Sum of Error vector = %i\n", errorsum);

    // Calculate the elapsed time
    float gputime;
    HIP_ERRCHK(hipSetDevice(0));
    HIP_ERRCHK(hipEventElapsedTime(&gputime, start, stop));
    printf("Time elapsed: %f\n", gputime / 1000.);
    for (int i = 0; i < 2; ++i)
    {
        HIP_ERRCHK(hipSetDevice(i));
        HIP_ERRCHK(hipEventElapsedTime(&gputime, start_h2d_event[i], stop_h2d_event[i]));
        printf("H2D on device %d: %f\n", i, gputime / 1000.);
        HIP_ERRCHK(hipEventElapsedTime(&gputime, start_kernel_event[i], stop_kernel_event[i]));
        printf("Kernel on device %d: %f\n", i, gputime / 1000.);
        HIP_ERRCHK(hipEventElapsedTime(&gputime, start_d2h_event[i], stop_d2h_event[i]));
        printf("D2H on device %d: %f\n", i, gputime / 1000.);
        HIP_ERRCHK(hipEventElapsedTime(&gputime, start_sync_event[i], stop_sync_event[i]));
        printf("SYNC on device %d: %f\n", i, gputime / 1000.);
        HIP_ERRCHK(hipEventDestroy(start_h2d_event[i]));
        HIP_ERRCHK(hipEventDestroy(start_kernel_event[i]));
        HIP_ERRCHK(hipEventDestroy(start_d2h_event[i]));
        HIP_ERRCHK(hipEventDestroy(start_sync_event[i]));
        HIP_ERRCHK(hipEventDestroy(stop_h2d_event[i]));
        HIP_ERRCHK(hipEventDestroy(stop_kernel_event[i]));
        HIP_ERRCHK(hipEventDestroy(stop_d2h_event[i]));
        HIP_ERRCHK(hipEventDestroy(stop_sync_event[i]));
    }

    HIP_ERRCHK(hipSetDevice(0));
    // Deallocate host memory
    HIP_ERRCHK(hipHostFree((void*)hA));
    HIP_ERRCHK(hipHostFree((void*)hB));
    HIP_ERRCHK(hipHostFree((void*)hC));

    return 0;
}
