#include <algorithm>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "common.h"


typedef uint32_t code_t;


template <class T>
class RadixTreeNode {
public:
    RadixTreeNode *left, *right, *parent;
    T element;
    __host__ __device__ bool isLeaf() { return left == nullptr && right == nullptr; }
};


template <class T, class CodeGetter>
class RadixTree {
public:
    RadixTreeNode<T> *root;
    __host__ __device__ void init(int count);
    __host__ __device__ void construct(T *buf);
    __host__ __device__ void destroy();
    __host__ void print();
    __host__ __device__ bool check() { return check(root, 0, count - 1); }
private:
    int count;
    RadixTreeNode<T> *internals, *leaves;
    __host__ __device__ int lcp(const T &a, int i, const T &b, int j);
    __host__ __device__ int lcp(int i, int j, T *buf);
    __host__ __device__ bool check(RadixTreeNode<T> *p, int l, int r);
    __host__ void printNode(RadixTreeNode<T> *p);
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
int RadixTree<T, CodeGetter>::lcp(const T &a, int i, const T &b, int j) {
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
int RadixTree<T, CodeGetter>::lcp(int i, int j, T *buf) {
    return lcp(buf[i], i, buf[j], j);
}


template <class T, class CodeGetter>
void RadixTree<T, CodeGetter>::printNode(RadixTreeNode<T> *p) {
    if (p->isLeaf()) {
        printf("L%d", int(p - leaves));
    }
    else {
        printf("I%d", int(p - internals));
    }
}


template <class T, class CodeGetter>
void RadixTree<T, CodeGetter>::print() {
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
bool RadixTree<T, CodeGetter>::check(RadixTreeNode<T> *p, int l, int r) {
    if (p == nullptr) return false;
    if (p->isLeaf()) {
        return l == r;
    }
    int split = l;
    int lcp_this = lcp(leaves[l].element, l, leaves[r].element, r);
    for (; split < r - 1 && lcp(leaves[l].element, l, leaves[split + 1].element, split + 1) > lcp_this; ++split);
    return check(p->left, l, split) && check(p->right, split + 1, r);
}
