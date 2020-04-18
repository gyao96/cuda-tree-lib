#include <algorithm>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include "common.h"


typedef uint32_t code_t;


template <class T>
class RadixTreeNode {
public:
    RadixTreeNode *left, *right, *parent;
    T element;
    __host__ __device__ bool isLeaf() const { return left == nullptr && right == nullptr; }
};


template <class T, class CodeGetter>
class RadixTree {
public:
    RadixTreeNode<T> *root;
    __host__ __device__ void init(int count);
    __host__ __device__ void init(int count, RadixTreeNode<T> *internals, RadixTreeNode<T> *leaves);
    __host__ __device__ void construct(T *buf);
    __host__ __device__ void destroy();
    __host__ void print() const;
    __host__ __device__ bool check() const { return check(root, 0, count - 1); }
private:
    int count;
    RadixTreeNode<T> *internals, *leaves;
    __host__ __device__ int lcp(const T &a, int i, const T &b, int j) const;
    __host__ __device__ int lcp(int i, int j, T *buf) const;
    __host__ __device__ bool check(const RadixTreeNode<T> *p, int l, int r) const;
    __host__ void printNode(const RadixTreeNode<T> *p) const;
};


template <class T, class CodeGetter>
void RadixTree<T, CodeGetter>::init(int count) {
#ifdef __CUDA_ARCH__
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
#else
    int tid = 0;
#endif
    if (tid == 0) {
        this->count = count;
        internals = new RadixTreeNode<T>[count - 1];
        leaves = new RadixTreeNode<T>[count];
        root = &internals[0];
        root->parent = nullptr;
    }
}


template <class T, class CodeGetter>
void RadixTree<T, CodeGetter>::init(int count, RadixTreeNode<T> *internals, RadixTreeNode<T> *leaves) {
    this->count = count;
    this->internals = internals;
    this->leaves = leaves;
    root = &internals[0];
    root->parent = nullptr;
}


template <class T, class CodeGetter>
void RadixTree<T, CodeGetter>::destroy() {
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


template <class T, class CodeGetter>
void RadixTree<T, CodeGetter>::construct(T *buf) {
#ifdef __CUDA_ARCH__
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int delta = blockDim.x * gridDim.x;
#else
    int tid = 0;
    int delta = 1;
#endif

    for (int i = tid; i < count; i += delta) {
        leaves[i].element = buf[i];
        leaves[i].left = leaves[i].right = nullptr;
    }

    for (int i = tid; i < count - 1; i += delta) {
        int d = (i == 0 ? 1 : lcp(i, i + 1, buf) - lcp(i, i - 1, buf));
        d = (d > 0 ? 1 : -1);
        int lcp_this = (i == 0 ? 0 : lcp(i, i - d, buf));

        // binary search the other end
        int beg, end;
        if (d == 1) {
            int l = i, r = count - 1;
            while (l < r) {
                int mid = (l + r + 1) / 2;
                if (lcp(i, mid, buf) < lcp_this) {
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
                if (lcp(i, mid, buf) < lcp_this) {
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
        lcp_this = lcp(beg, end, buf);
        int l = beg, r = end - 1;
        while (l < r) {
            int mid = (l + r + 1) / 2;
            if (lcp(beg, mid, buf) == lcp_this) {
                r = mid - 1;
            }
            else {
                l = mid;
            }
        }
        int split = l;

        RadixTreeNode<T> *left = (split == beg ? &leaves[split] : &internals[split]);
        RadixTreeNode<T> *right = (split == end - 1 ? &leaves[split + 1] : &internals[split + 1]);
        internals[i].left = left;
        internals[i].right = right;
        left->parent = right->parent = &internals[i];
    }
}


template <class T, class CodeGetter>
int RadixTree<T, CodeGetter>::lcp(const T &a, int i, const T &b, int j) const {
    uint64_t x = ((uint64_t)CodeGetter()(a) << 32) | i;
    uint64_t y = ((uint64_t)CodeGetter()(b) << 32) | j;
#ifdef __CUDA_ARCH__
    return __clzll(x ^ y);
#else
    int res = 64;
    for (int z = x ^ y; z > 0; z >>= 1, --res);
    return res;
#endif
}

template <class T, class CodeGetter>
int RadixTree<T, CodeGetter>::lcp(int i, int j, T *buf) const {
    return lcp(buf[i], i, buf[j], j);
}


template <class T, class CodeGetter>
void RadixTree<T, CodeGetter>::printNode(const RadixTreeNode<T> *p) const {
    if (p->isLeaf()) {
        printf("L%d", int(p - leaves));
    }
    else {
        printf("I%d", int(p - internals));
    }
}


template <class T, class CodeGetter>
void RadixTree<T, CodeGetter>::print() const {
    printf("Tree: %d elements\n", count);
    for (int i = 0; i < count - 1; ++i) {
        printf("\t");
        printNode(&internals[i]);
        printf(": ");
        printNode(internals[i].left);
        printf(" ");
        printNode(internals[i].right);
        printf("\n");
    }
    for (int i = 0; i < count; ++i) {
        printf("\t");
        printNode(&leaves[i]);
    }
}


template <class T, class CodeGetter>
bool RadixTree<T, CodeGetter>::check(const RadixTreeNode<T> *p, int l, int r) const {
    if (p == nullptr) return false;
    if (p->isLeaf()) {
        return l == r;
    }
    int split = l;
    int lcp_this = lcp(leaves[l].element, l, leaves[r].element, r);
    for (; split < r - 1 && lcp(leaves[l].element, l, leaves[split + 1].element, split + 1) > lcp_this; ++split);
    return check(p->left, l, split) && check(p->right, split + 1, r);
}



// Wrapper for CUDA version

template <class T, class CodeGetter>
struct Comp {
    __host__ __device__ bool operator()(const T &a, const T &b) {
        return CodeGetter()(a) < CodeGetter()(b);
    }
};

template <class T, class CodeGetter>
__global__ void _init(RadixTree<T, CodeGetter> *tree, int count,
        RadixTreeNode<T> *internals, RadixTreeNode<T> *leaves) {
    tree->init(count, internals, leaves);
}

template <class T, class CodeGetter>
__global__ void _construct(RadixTree<T, CodeGetter> *tree, T *buf) {
    tree->construct(buf);
}

template <class T, class CodeGetter>
__global__ void _check(const RadixTree<T, CodeGetter> *tree, bool *res) {
    *res = tree->check();
}

template <class T, class CodeGetter>
__global__ void _print(const RadixTree<T, CodeGetter> *tree) {
    tree->print();
}

#define N_THREADS_PER_BLK 256

template <class T, class CodeGetter>
class RadixTreeWrapper {
public:
    RadixTreeWrapper(int count) : count(count), data_dev(nullptr),
            internals_dev(nullptr), leaves_dev(nullptr), tree_dev(nullptr) {
        cudaMalloc(&tree_dev, sizeof(RadixTree<T, CodeGetter>));
        cudaMalloc(&data_dev, count * sizeof(T));
        cudaMalloc(&internals_dev, (count - 1) * sizeof(RadixTreeNode<T>));
        cudaMalloc(&leaves_dev, count * sizeof(RadixTreeNode<T>));
        _init<<<1, 1>>>(tree_dev, count, internals_dev, leaves_dev);
        cudaDeviceSynchronize();
    }
    ~RadixTreeWrapper() {
        if (data_dev)
            cudaFree(data_dev);
        if (internals_dev)
            cudaFree(internals_dev);
        if (leaves_dev)
            cudaFree(leaves_dev);
        if (tree_dev)
            cudaFree(tree_dev);
    }
    void construct(T *buf) {
        thrust::copy(thrust::device_ptr<T>(buf), thrust::device_ptr<T>(buf) + count, thrust::device_ptr<T>(data_dev));
        thrust::sort(thrust::device_ptr<T>(data_dev), thrust::device_ptr<T>(data_dev) + count, Comp<T, CodeGetter>());
        int nblks = min(64, (count + N_THREADS_PER_BLK - 1) / N_THREADS_PER_BLK);
        _construct<<<nblks, N_THREADS_PER_BLK>>>(tree_dev, data_dev);
        cudaDeviceSynchronize();
    }
    bool check() const {
        bool *res_dev;
        cudaMalloc(&res_dev, sizeof(bool));
        _check<<<1, 1>>>(tree_dev, res_dev);
        bool res;
        cudaMemcpy(&res, res_dev, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(res_dev);
        cudaDeviceSynchronize();
        return res;
    }
    void print() const { _print<<<1, 1>>>(tree_dev); }
    RadixTree<T, CodeGetter> *tree() { return tree_dev; }
private:
    int count;
    RadixTree<T, CodeGetter> *tree_dev;
    T *data_dev;
    RadixTreeNode<T> *internals_dev, *leaves_dev;
};
