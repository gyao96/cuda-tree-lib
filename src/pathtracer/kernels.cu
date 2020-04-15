#include <cuda.h>
#include <stdio.h>
#include "kernels.h"
namespace CGL
{

    __device__ int addem(int a, int b)
    {
        return a + b;
    }

    __global__ void add(int a, int b, int* c)
    {
        *c = addem(a, b);
    }


    void pathtraceInit()
    {

        int c;
        int* dev_c;
        cudaMalloc((void**)&dev_c, sizeof(int));

        add << <1, 1 >> > (2, 7, dev_c);

        cudaMemcpy(&c, dev_c, sizeof(int),
            cudaMemcpyDeviceToHost);
        printf("2 + 7 = %d\n", c);
        cudaFree(dev_c);

    }
}
