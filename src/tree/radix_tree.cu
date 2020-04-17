#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iostream>


class RadixTreeNode {
public:
    RadixTreeNode *left, *right, *parent;
    int val;
    bool isLeaf() { return left == nullptr && right == nullptr; }
};


class RadixTree {
public:
    RadixTreeNode *root;
    RadixTree(int *buf, int count);
    ~RadixTree();
    void insert(int x);
    void construct(int *buf);
    void print();
    bool check() { return check(root, 0, count - 1); }
private:
    int count;
    RadixTreeNode *internals, *leaves;
    int lcp(int x, int y);
    bool check(RadixTreeNode *p, int l, int r);
    void printNode(RadixTreeNode *p);
};


RadixTree::RadixTree(int *buf, int count) : count(count) {
    internals = new RadixTreeNode[count - 1];
    leaves = new RadixTreeNode[count];
    for (int i = 0; i < count; ++i) {
        leaves[i].val = buf[i];
        leaves[i].left = leaves[i].right = nullptr;
    }
    internals[0].parent = nullptr;
    construct(buf);
}


RadixTree::~RadixTree() {
    delete[] internals;
    delete[] leaves;
}


void RadixTree::construct(int *buf) {
    root = &internals[0];
    for (int i = 0; i < count - 1; ++i) {
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
    int res = 32;
    for (int z = x ^ y; z > 0; z >>= 1, --res);
    return res;
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
    std::cout << "Tree:" << std::endl;
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


const int N = 100000, MAX = 10000000;
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
    /*
    for (int i = 0; i < n; ++i) std::cout << arr[i] << " ";
    std::cout << std::endl;
    */

    RadixTree tree(arr, n);
    // tree.print();
    std::cout << (tree.check() ? "Success" : "Failed") << std::endl;
}
