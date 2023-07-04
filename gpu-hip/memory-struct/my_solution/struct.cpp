#include <cstdio>
#include <hip/hip_runtime.h>

/* Blocksize divisible by the warp size */
#define BLOCKSIZE 64

/* Example struct to practise copying structs with pointers to device memory */
typedef struct
{
  float *x;
  int *idx;
  int size;
} Example;

/* GPU kernel definition */
__global__ void hipKernel(Example* const d_ex)
{
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread < d_ex->size)
  {
    printf("x[%d]: %.2f, idx[%d]:%d/%d \n", thread, d_ex->x[thread], thread, d_ex->idx[thread], d_ex->size - 1);
  }
}

/* Run on host */
void runHost()
{
  // Allocate host struct
  Example *ex;
  ex = (Example*)malloc(sizeof(Example));
  ex->size = 10;

  // Allocate host struct members
  ex->x = (float*)malloc(ex->size * sizeof(float));
  ex->idx = (int*)malloc(ex->size * sizeof(int));

  // Initialize host struct members
  for(int i = 0; i < ex->size; i++)
  {
    ex->x[i] = (float)i;
    ex->idx[i] = i;
  }

  // Print struct values from host
  printf("\nHost:\n");
  for(int i = 0; i < ex->size; i++)
  {
    printf("x[%d]: %.2f, idx[%d]:%d/%d \n", i, ex->x[i], i, ex->idx[i], ex->size - 1);
  }

  // Free host struct
  free(ex->x);
  free(ex->idx);
  free(ex);
}

/* Run on device using Unified Memory */
void runDeviceUnifiedMem()
{
  // Allocate struct using Unified Memory
  Example *ex;
  hipMallocManaged(&ex, sizeof(Example));
  ex->size = 10;

  // Allocate struct members using Unified Memory
  hipMallocManaged(&ex->x, sizeof(float) * ex->size);
  hipMallocManaged(&ex->idx, sizeof(int) * ex->size);

  // Initialize struct from host
  for(int i = 0; i < ex->size; i++)
  {
    ex->x[i] = (float)i;
    ex->idx[i] = i;
  }

  // Print struct values from device by calling hipKernel()
  printf("\nDevice (UnifiedMem):\n");
  int gridsize = (ex->size - 1 + BLOCKSIZE) / BLOCKSIZE;
  hipLaunchKernelGGL(hipKernel, gridsize, BLOCKSIZE, 0, 0, ex);
  hipStreamSynchronize(0);

  // Free struct
  hipFree(ex->x);
  hipFree(ex->idx);
  hipFree(ex);
}

/* Create the device struct (needed for explicit memory management) */
Example* createDeviceExample(Example *ex)
{
  // Allocate device struct
  Example *ex_d;
  hipMalloc(&ex_d, sizeof(Example));

  // Allocate device struct members
  hipMalloc(&(ex_d->x), sizeof(float) * ex->size);
  hipMalloc(&(ex_d->idx), sizeof(int) * ex->size);

  // Copy arrays pointed by the struct members from host to device
  hipMemcpy(ex_d->x, ex->x, ex->size * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(ex_d->idx, ex->idx, ex->size * sizeof(int), hipMemcpyHostToDevice);

  // Copy struct members from host to device
  hipMemcpy(&ex_d->size, &ex->size, sizeof(int), hipMemcpyHostToDevice);

  // Return device struct
  return ex_d;
}

/* Free the device struct (needed for explicit memory management) */
void freeDeviceExample(Example *ex_d)
{
  // #error Copy struct members (pointers) from device to host

  // Free device struct members
  hipFree(ex_d->x);
  hipFree(ex_d->idx);

  // Free device struct
  hipFree(ex_d);
}

/* Run on device using Explicit memory management */
void runDeviceExplicitMem()
{
  // Allocate host struct
  Example *ex = new Example;
  ex->size = 10;

  // Allocate host struct members
  ex->x = new float [ex->size];
  ex->idx = new int [ex->size];

  // Initialize host struct
  for(int i = 0; i < ex->size; i++)
  {
    ex->x[i] = (float)i;
    ex->idx[i] = i;
  }

  // Allocate device struct and copy values from host to device
  Example *ex_d = createDeviceExample(ex);

  // Print struct values from device by calling hipKernel()
  printf("\nDevice (ExplicitMem):\n");
  int gridsize = (ex->size - 1 + BLOCKSIZE) / BLOCKSIZE;
  hipLaunchKernelGGL(hipKernel, gridsize, BLOCKSIZE, 0, 0, ex_d);
  hipStreamSynchronize(0);

  // Free device struct
  freeDeviceExample(ex_d);

  // Free host struct
  delete [] ex->x;
  delete [] ex->idx;
  delete ex;
}

/* The main function */
int main(int argc, char* argv[])
{
  runHost();
  runDeviceUnifiedMem();
  runDeviceExplicitMem();
}
