#include <type_traits>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include "common.h"


typedef uint32_t code_t;


template <class T, class V>
class RadixTreeNode {
public:
    RadixTreeNode *left, *right, *parent;
    T element;
    V value;
    __host__ __device__ bool isLeaf() const { return left == nullptr && right == nullptr; }
};


template <class Tree, class T, class V> class RadixTreeWrapper;

template <class T, class V>
class RadixTreeGeneric {
public:
    RadixTreeNode<T, V> *root;
    __host__ __device__ void init(int count);
    __host__ __device__ void init(int count, RadixTreeNode<T, V> *internals, RadixTreeNode<T, V> *leaves);
    __host__ __device__ void construct(T *buf);
    __host__ __device__ void populateValue();
    __host__ __device__ void destroy();
    __host__ __device__ void generateCodesFor(int n, T *src, code_t *dst) const;
    __host__ __device__ void print() const;
    __host__ __device__ bool check() const { return check(root, 0, count - 1); }
    friend class RadixTreeWrapper<RadixTreeGeneric<T, V>, T, V>;
private:
    int count;
    RadixTreeNode<T, V> *internals, *leaves;
    __host__ __device__ int lcp(const T &a, int i, const T &b, int j) const;
    __host__ __device__ int lcp(int i, int j, T *buf) const;
    __host__ __device__ bool check(const RadixTreeNode<T, V> *p, int l, int r) const;
    __host__ __device__ void printNode(const RadixTreeNode<T, V> *p) const;
protected:
    __host__ __device__ virtual code_t getCode(const T &x) const = 0;
    __host__ __device__ virtual V startValue() const = 0;
    __host__ __device__ virtual V elementToValue(const T &x) const = 0;
    __host__ __device__ virtual void update(V &dst, const V &src) const = 0;
    __device__ virtual void atomicUpdate(V &dst, const V &src) const { update(dst, src); }
};


template <class T, class V>
void RadixTreeGeneric<T, V>::init(int count) {
#ifdef __CUDA_ARCH__
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
#else
    int tid = 0;
#endif
    if (tid == 0) {
        this->count = count;
        internals = new RadixTreeNode<T, V>[count - 1];
        leaves = new RadixTreeNode<T, V>[count];
        root = &internals[0];
        root->parent = nullptr;
    }
}


template <class T, class V>
void RadixTreeGeneric<T, V>::init(int count, RadixTreeNode<T, V> *internals, RadixTreeNode<T, V> *leaves) {
    this->count = count;
    this->internals = internals;
    this->leaves = leaves;
    root = &internals[0];
    root->parent = nullptr;
}


template <class T, class V>
void RadixTreeGeneric<T, V>::destroy() {
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


template <class T, class V>
void RadixTreeGeneric<T, V>::generateCodesFor(int n, T *src, code_t *dst) const {
#ifdef __CUDA_ARCH__
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int delta = blockDim.x * gridDim.x;
#else
    int tid = 0;
    int delta = 1;
#endif

    for (int i = tid; i < n; i += delta) {
        dst[i] = getCode(src[i]);
    }
}


template <class T, class V>
void RadixTreeGeneric<T, V>::construct(T *buf) {
#ifdef __CUDA_ARCH__
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int delta = blockDim.x * gridDim.x;
#else
    int tid = 0;
    int delta = 1;
#endif

    for (int i = tid; i < count; i += delta) {
        leaves[i].element = buf[i];
        leaves[i].value = elementToValue(leaves[i].element);
        leaves[i].left = leaves[i].right = nullptr;
    }

    for (int i = tid; i < count - 1; i += delta) {
        internals[i].value = startValue();
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

        RadixTreeNode<T, V> *left = (split == beg ? &leaves[split] : &internals[split]);
        RadixTreeNode<T, V> *right = (split == end - 1 ? &leaves[split + 1] : &internals[split + 1]);
        internals[i].left = left;
        internals[i].right = right;
        left->parent = right->parent = &internals[i];
    }
}


template <class T, class V>
void RadixTreeGeneric<T, V>::populateValue() {
#ifdef __CUDA_ARCH__
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int delta = blockDim.x * gridDim.x;
#else
    int tid = 0;
    int delta = 1;
#endif

    for (int i = tid; i < count; i += delta) {
        for (RadixTreeNode<T, V> *p = leaves[i].parent; p != nullptr; p = p->parent) {
#ifdef __CUDA_ARCH__
            atomicUpdate(p->value, leaves[i].value);
#else
            update(p->value, leaves[i].value);
#endif
        }
    }
}


template <class T, class V>
int RadixTreeGeneric<T, V>::lcp(const T &a, int i, const T &b, int j) const {
    uint64_t x = ((uint64_t)getCode(a) << 32) | i;
    uint64_t y = ((uint64_t)getCode(b) << 32) | j;
#ifdef __CUDA_ARCH__
    return __clzll(x ^ y);
#else
    int res = 64;
    for (int z = x ^ y; z > 0; z >>= 1, --res);
    return res;
#endif
}

template <class T, class V>
int RadixTreeGeneric<T, V>::lcp(int i, int j, T *buf) const {
    return lcp(buf[i], i, buf[j], j);
}


template <class T, class V>
void RadixTreeGeneric<T, V>::printNode(const RadixTreeNode<T, V> *p) const {
    if (p->isLeaf()) {
        printf("L%d", int(p - leaves));
    }
    else {
        printf("I%d", int(p - internals));
    }
}


template <class T, class V>
void RadixTreeGeneric<T, V>::print() const {
    printf("Tree: %d elements\n", count);
    for (int i = 0; i < count - 1; ++i) {
        printf("\t");
        printNode(&internals[i]);
        printf("(%d): ", internals[i].value);
        printNode(internals[i].left);
        printf(" ");
        printNode(internals[i].right);
        printf("\n");
    }
    for (int i = 0; i < count; ++i) {
        printf("\t");
        printNode(&leaves[i]);
        printf("(%d)\n", leaves[i].value);
    }
}


template <class T, class V>
bool RadixTreeGeneric<T, V>::check(const RadixTreeNode<T, V> *p, int l, int r) const {
    if (p == nullptr) return false;
    if (p->isLeaf()) {
        return l == r;
    }
    int split = l;
    int lcp_this = lcp(leaves[l].element, l, leaves[r].element, r);
    for (; split < r - 1 && lcp(leaves[l].element, l, leaves[split + 1].element, split + 1) > lcp_this; ++split);
    V v = p->left->value;
    update(v, p->right->value);
    if (v != p->value) return false;
    return check(p->left, l, split) && check(p->right, split + 1, r);
}



// Wrapper for CUDA version

template <class Tree, class T, class V>
__global__ void _init(Tree **tree_ptr, int count,
        RadixTreeNode<T, V> *internals, RadixTreeNode<T, V> *leaves) {
    *tree_ptr = new Tree();
    (*tree_ptr)->init(count, internals, leaves);
}

template <class Tree>
__global__ void _destroy(Tree *tree) {
    delete tree;
}

template <class Tree, class T>
__global__ void _construct(Tree *tree, T *buf) {
    tree->construct(buf);
}

template <class Tree, class T>
__global__ void _generateCodesFor(Tree *tree, int n, T *src, code_t *dst) {
    tree->generateCodesFor(n, src, dst);
}

template <class Tree>
__global__ void _populateValue(Tree *tree) {
    tree->populateValue();
}

template <class Tree>
__global__ void _check(const Tree *tree, bool *res) {
    *res = tree->check();
}

template <class Tree>
__global__ void _print(const Tree *tree) {
    tree->print();
}

#define N_THREADS_PER_BLK 256


template <class Tree,
          class T = typename std::decay<decltype(Tree::root->element)>::type,
          class V = typename std::decay<decltype(Tree::root->value)>::type>
class RadixTreeWrapper {
public:
    RadixTreeWrapper(int count) : count(count), data_dev(nullptr),
            internals_dev(nullptr), leaves_dev(nullptr), tree_dev(nullptr) {
        cudaMalloc(&data_dev, count * sizeof(T));
        cudaMalloc(&codes_dev, count * sizeof(code_t));
        cudaMalloc(&internals_dev, (count - 1) * sizeof(RadixTreeNode<T, V>));
        cudaMalloc(&leaves_dev, count * sizeof(RadixTreeNode<T, V>));
        Tree **tree_dev_ptr;
        cudaMalloc(&tree_dev_ptr, sizeof(Tree *));
        _init<<<1, 1>>>(tree_dev_ptr, count, internals_dev, leaves_dev);
        cudaMemcpy(&tree_dev, tree_dev_ptr, sizeof(Tree *), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaFree(tree_dev_ptr);
    }
    ~RadixTreeWrapper() {
        cudaFree(data_dev);
        cudaFree(codes_dev);
        cudaFree(internals_dev);
        cudaFree(leaves_dev);
        _destroy<<<1, 1>>>(tree_dev);
    }
    void construct(T *buf) {
        int nblks = min(64, (count + N_THREADS_PER_BLK - 1) / N_THREADS_PER_BLK);
        thrust::copy(thrust::device_ptr<T>(buf), thrust::device_ptr<T>(buf) + count, thrust::device_ptr<T>(data_dev));
        _generateCodesFor<<<nblks, N_THREADS_PER_BLK>>>(tree_dev, count, data_dev, codes_dev);
        thrust::sort_by_key(thrust::device_ptr<code_t>(codes_dev), thrust::device_ptr<code_t>(codes_dev) + count,
                thrust::device_ptr<T>(data_dev));
        _construct<<<nblks, N_THREADS_PER_BLK>>>(tree_dev, data_dev);
        _populateValue<<<nblks, N_THREADS_PER_BLK>>>(tree_dev);
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
    Tree *tree() { return tree_dev; }
private:
    int count;
    Tree *tree_dev;
    T *data_dev;
    code_t *codes_dev;
    RadixTreeNode<T, V> *internals_dev, *leaves_dev;
};
