#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "radix_tree_generic.h"

#define N_THREADS_PER_BLK 256


class RadixTree : public RadixTreeGeneric<int, int> {
protected:
    __host__ __device__ code_t getCode(const int &x) const override {
        return x;
    }
    __host__ __device__ int startValue() const override {
        return 0;
    }
    __host__ __device__ int elementToValue(const int &x) const override {
        return x;
    }
    __host__ __device__ void update(int &dst, const int &src) const override {
        dst ^= src;
    }
    __device__ void atomicUpdate(int &dst, const int &src) const override {
        atomicXor(&dst, src);
    }
};


__global__  void init(RadixTree *tree, int n) {
    tree->init(n);
}
__global__ void construct(RadixTree *tree, int *arr) {
    tree->construct(arr);
}
__global__ void destroy(RadixTree *tree) {
    tree->destroy();
}
__global__ void check(RadixTree *tree, bool *res) {
    *res = tree->check();
}


const int N = 10, MAX = 100;
int arr[N];

int main() {
    srand(time(0));
    for (int i = 0; i < N; ++i) arr[i] = rand() % MAX;
    std::sort(arr, arr + N);
    int n = 0;
    for (int i = 0; i < N; ++i)
        if (i == 0 || arr[i] != arr[i - 1])
            arr[n++] = arr[i];
    std::random_shuffle(arr, arr + n);
    std::cout << n << std::endl;

    bool res = false;

    // CPU version
    /*
    std::sort(arr, arr + n);
    RadixTree tree;
    tree.init(n);
    tree.construct(arr);
    res = tree.check();
    tree.print();
    tree.destroy();
    */

    // GPU version
    int *arr_dev;
    bool *res_dev;
    RadixTree *tree_dev;
    cudaMalloc(&arr_dev, n * sizeof(int));
    cudaMalloc(&res_dev, sizeof(bool));
    cudaMalloc(&tree_dev, sizeof(RadixTree));
    cudaMemcpy(arr_dev, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    /*
    init<<<1, 1>>>(tree_dev, n);
    thrust::sort(thrust::device_ptr<int>(arr_dev), thrust::device_ptr<int>(arr_dev) + n);
    int nblks = min(64, (n + N_THREADS_PER_BLK - 1) / N_THREADS_PER_BLK);
    construct<<<nblks, N_THREADS_PER_BLK>>>(tree_dev, arr_dev);
    check<<<1, 1>>>(tree_dev, res_dev);
    cudaMemcpy(&res, res_dev, sizeof(bool), cudaMemcpyDeviceToHost);
    destroy<<<1, 1>>>(tree_dev);
    */

    RadixTreeWrapper<RadixTree> tw(n);
    tw.construct(arr_dev);
    tw.print();
    res = tw.check();

    cudaDeviceSynchronize();
    cudaCheckError();
    cudaFree(arr_dev);
    cudaFree(res_dev);
    cudaFree(tree_dev);

    std::cout << (res ? "Success" : "Failed") << std::endl;
    return 0;
}
