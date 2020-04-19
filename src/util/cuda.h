#ifndef CUDA_H
#define CUDA_H

#ifdef __NVCC__
#define __QUALIFIER__ __host__ __device__
#else
#define __QUALIFIER__
#endif

#endif
