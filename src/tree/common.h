#ifndef __TREE_COMMON_H__
#define __TREE_COMMON_H__

#include <iostream>

#define cudaCheckError() {  \
    cudaError_t e = cudaGetLastError();  \
    if (e != cudaSuccess) {  \
        std::cout << "Cuda error in " << std::string(__FILE__) << " line " << __LINE__ << ": " << cudaGetErrorString(e) << std::endl;  \
        exit(0);  \
    }  \
}

#endif
