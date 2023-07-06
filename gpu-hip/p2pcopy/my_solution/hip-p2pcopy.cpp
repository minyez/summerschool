#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <hip/hip_runtime.h>


void copyP2P(int p2p, int gpu0, int gpu1, int* dA_0, int* dA_1, int size) {

    unsigned int flag;
    // Enable peer access for GPUs?
    if (p2p)
    {
        // Enable peer access for GPU 0 and GPU 1
        hipSetDevice(gpu0);
        hipDeviceEnablePeerAccess(gpu1, flag);
        hipSetDevice(gpu1);
        hipDeviceEnablePeerAccess(gpu0, flag);
    }

    // Do a dummy copy without timing to remove the impact of the first one
    // Copy dA_1 on device 1 to dA_0 on device 0
    hipMemcpyPeer(dA_0, gpu0, dA_1, gpu1, size);
    // hipMemcpy(dA_0, dA_1, size, hipMemcpyDefault);

    // Do a series of timed P2P memory copies
    int N = 10;
    clock_t tStart = clock();
    // Copy dA_1 on device 1 to dA_0 on device 0, repeat for N times to
    // get timings
    for (int i = 0; i < N; i++)
    {
        hipMemcpyPeer(dA_0, gpu0, dA_1, gpu1, size);
        // hipMemcpy(dA_0, dA_1, size, hipMemcpyDefault);
    }

    // After the memory copies, remember to synchronize the stream
    //       before stopping the clock
    // NOTE: no need to synchronize in this case, as hipMemcpy is blocking
    //       In general case, it is either not needed to synchronize both src and dst.
    //       Sync on all devices is safe to do, but introduce unnecessary overhead.
    // hipSetDevice(gpu0);
    // hipStreamSynchronize(0);
    // hipSetDevice(gpu1);
    // hipStreamSynchronize(0);
    clock_t tStop = clock();

    // Calcute time and bandwith
    double time_s = (double) (tStop - tStart) / CLOCKS_PER_SEC;
    double bandwidth = (double) size * (double) N / (double) 1e9 / time_s;

    // Disable peer access for GPUs?
    if (p2p) {
        // TODO: Disable peer access for GPU 0 and GPU 1
        hipSetDevice(gpu0);
        hipDeviceDisablePeerAccess(gpu1);
        hipSetDevice(gpu1);
        hipDeviceDisablePeerAccess(gpu0);
        printf("P2P enabled - Bandwith: %.3f (GB/s), Time: %.3f s\n",
                bandwidth, time_s);
    } else {
        printf("P2P disabled - Bandwith: %.3f (GB/s), Time: %.3f s\n",
                bandwidth, time_s);
    }
}


int main(int argc, char *argv[])
{
    // Check that we have at least two GPUs
    int devcount;
    hipGetDeviceCount(&devcount);
    if(devcount < 2) {
        printf("Need at least two GPUs!\n");
        exit(EXIT_FAILURE);
    } else {
        printf("Found %d GPU devices, using GPUs 0 and 1!\n", devcount);
    }

    // Allocate memory for both GPUs
    int size = pow(2, 28);
    int gpu0 = 0, gpu1 = 1;
    int *dA_0, *dA_1;
    hipSetDevice(gpu0);
    hipMalloc((void**) &dA_0, size);
    hipSetDevice(gpu1);
    hipMalloc((void**) &dA_1, size);

    // Check peer accessibility between GPUs 0 and 1
    int peerAccess01;
    int peerAccess10;
    // Check for peer to peer accessibility from device 0 to 1
    hipSetDevice(gpu0);
    hipDeviceCanAccessPeer(&peerAccess01, gpu0, gpu1);
    hipSetDevice(gpu1);
    hipDeviceCanAccessPeer(&peerAccess10, gpu1, gpu0);
    //       and from 1 to 0
    printf("hipDeviceCanAccessPeer: %d (GPU %d to GPU %d)\n",
            peerAccess01, gpu0, gpu1);
    printf("hipDeviceCanAccessPeer: %d (GPU %d to GPU %d)\n",
            peerAccess10, gpu1, gpu0);

    // Memcopy, P2P enabled
    if (peerAccess01 && peerAccess10)
        copyP2P(1, gpu0, gpu1, dA_0, dA_1, size);

    // Memcopy, P2P disabled
    copyP2P(0, gpu0, gpu1, dA_0, dA_1, size);

    // Deallocate device memory
    hipFree(dA_0);
    hipFree(dA_1);
}