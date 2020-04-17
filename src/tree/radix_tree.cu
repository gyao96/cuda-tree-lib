#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iostream>
#include "common.h"

#define N_THREADS_PER_BLK 256


class RadixTreeNode {
public:
    RadixTreeNode *left, *right, *parent;
    int val;
    __host__ __device__ bool isLeaf() { return left == nullptr && right == nullptr; }
};


class RadixTree {
public:
    RadixTreeNode *root;
    __host__ __device__ void init(int count);
    __host__ __device__ void construct(int *buf);
    __host__ __device__ void destroy();
    __host__ void print();
    __host__ __device__ bool check() { return check(root, 0, count - 1); }
private:
    int count;
    RadixTreeNode *internals, *leaves;
    __host__ __device__ int lcp(int x, int y);
    __host__ __device__ bool check(RadixTreeNode *p, int l, int r);
    __host__ void printNode(RadixTreeNode *p);
};


void RadixTree::init(int count) {
#ifdef __CUDA_ARCH__
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
#else
    int tid = 0;
#endif
    if (tid == 0) {
        this->count = count;
        internals = new RadixTreeNode[count - 1];
        leaves = new RadixTreeNode[count];
        root = &internals[0];
        root->parent = nullptr;
    }
}


void RadixTree::destroy() {
#ifdef __CUDA_ARCH__
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
#else
    int tid = 0;
#endif
    if (tid == 0) {
        delete[] internals;
        delete[] leaves;
    }
}


void RadixTree::construct(int *buf) {
#ifdef __CUDA_ARCH__
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int delta = blockDim.x * gridDim.x;
#else
    int tid = 0;
    int delta = 1;
#endif

    for (int i = tid; i < count; i += delta) {
        leaves[i].val = buf[i];
        leaves[i].left = leaves[i].right = nullptr;
    }

    for (int i = tid; i < count - 1; i += delta) {
        int d = (i == 0 ? 1 : lcp(buf[i], buf[i + 1]) - lcp(buf[i], buf[i - 1]));
        d = (d > 0 ? 1 : -1);
        int lcp_this = (i == 0 ? 0 : lcp(buf[i], buf[i - d]));

        // binary search the other end
        int beg, end;
        if (d == 1) {
            int l = i, r = count - 1;
            while (l < r) {
                int mid = (l + r + 1) / 2;
                if (lcp(buf[i], buf[mid]) < lcp_this) {
                    r = mid - 1;
                }
                else {
                    l = mid;
                }
            }
            beg = i;
            end = l;
        }
        else {
            int l = 0, r = i;
            while (l < r) {
                int mid = (l + r) / 2;
                if (lcp(buf[i], buf[mid]) < lcp_this) {
                    l = mid + 1;
                }
                else {
                    r = mid;
                }
            }
            beg = l;
            end = i;
        }

        // binary search split point
        lcp_this = lcp(buf[beg], buf[end]);
        int l = beg, r = end - 1;
        while (l < r) {
            int mid = (l + r + 1) / 2;
            if (lcp(buf[beg], buf[mid]) == lcp_this) {
                r = mid - 1;
            }
            else {
                l = mid;
            }
        }
        int split = l;

        RadixTreeNode *left = (split == beg ? &leaves[split] : &internals[split]);
        RadixTreeNode *right = (split == end - 1 ? &leaves[split + 1] : &internals[split + 1]);
        internals[i].left = left;
        internals[i].right = right;
        left->parent = right->parent = &internals[i];
    }
}


int RadixTree::lcp(int x, int y) {
#ifdef __CUDA_ARCH__
    return __clz(x ^ y);
#else
    int res = 32;
    for (int z = x ^ y; z > 0; z >>= 1, --res);
    return res;
#endif
}


void RadixTree::printNode(RadixTreeNode *p) {
    if (p->isLeaf()) {
        std::cout << "L" << p - leaves;
    }
    else {
        std::cout << "I" << p - internals;
    }
}


void RadixTree::print() {
    std::cout << "Tree: " << count << " elements" << std::endl;
    for (int i = 0; i < count - 1; ++i) {
        std::cout << "\t";
        printNode(&internals[i]);
        std::cout << ": ";
        printNode(internals[i].left);
        std::cout << " ";
        printNode(internals[i].right);
        std::cout << std::endl;
    }
    for (int i = 0; i < count; ++i) {
        std::cout << "\t";
        printNode(&leaves[i]);
        std::cout << ": " << leaves[i].val << std::endl;
    }
}


bool RadixTree::check(RadixTreeNode *p, int l, int r) {
    if (p == nullptr) return false;
    if (p->isLeaf()) {
        return l == r;
    }
    int split = l;
    int lcp_this = lcp(leaves[l].val, leaves[r].val);
    for (; split < r - 1 && lcp(leaves[l].val, leaves[split + 1].val) > lcp_this; ++split);
    return check(p->left, l, split) && check(p->right, split + 1, r);
}


/*
__global__ void test(int *arr, int n, bool *res) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    RadixTree tree(arr, n);
    __syncthreads();
    if (tid == 0) {
        *res = tree.check();
    }
}
*/


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


const int N = 1000000, MAX = 10000000;
int arr[N];

int main() {
    srand(time(0));
    for (int i = 0; i < N; ++i) arr[i] = rand() % MAX;
    std::sort(arr, arr + N);
    int n = 0;
    for (int i = 0; i < N; ++i)
        if (i == 0 || arr[i] != arr[i - 1])
            arr[n++] = arr[i];
    std::cout << n << std::endl;

    bool res;

    // CPU version
    /*
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

    init<<<1, 1>>>(tree_dev, n);

    int nblks = min(64, (n + N_THREADS_PER_BLK - 1) / N_THREADS_PER_BLK);
    // test<<<nblks, N_THREADS_PER_BLK>>>(arr_dev, n, res_dev);
    construct<<<nblks, N_THREADS_PER_BLK>>>(tree_dev, arr_dev);
    check<<<1, 1>>>(tree_dev, res_dev);
    cudaMemcpy(&res, res_dev, sizeof(bool), cudaMemcpyDeviceToHost);

    destroy<<<1, 1>>>(tree_dev);
    cudaDeviceSynchronize();
    cudaCheckError();
    cudaFree(arr_dev);
    cudaFree(res_dev);

    std::cout << (res ? "Success" : "Failed") << std::endl;
    return 0;
}
