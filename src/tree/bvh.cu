#include "scene/bbox.h"
#include "scene/primitive.h"
#include "radix_tree_generic.h"

using namespace CGL;
using namespace CGL::SceneObjects;


__host__ __device__ inline uint32_t spaceBy3(uint32_t x) {
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x <<  8)) & 0x0300F00F;
    x = (x | (x <<  4)) & 0x030C30C3;
    x = (x | (x <<  2)) & 0x09249249;
    return x;
}

__device__ void atomicMin(double * const address, const double value) {
    if (*address <= value) return;
    unsigned long long * const address_as_i = (unsigned long long *)address;
    unsigned long long old = *address_as_i, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= value) break;
        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

__device__ void atomicMax(double * const address, const double value) {
    if (*address >= value) return;
    unsigned long long * const address_as_i = (unsigned long long *)address;
    unsigned long long old = *address_as_i, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= value) break;
        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while (assumed != old);
}


class BVHTree : public RadixTreeGeneric<Primitive *, BBox> {
public:
    __host__ __device__ BVHTree(Vector3D min = Vector3D(-1000, -1000, -1000),
            Vector3D max = Vector3D(1000, 1000, 1000)) : min(min), max(max) {}
protected:
    __host__ __device__ code_t getCode(Primitive * const &x) const override {
        Vector3D centroid = (x->get_bbox().centroid() - min) / (max - min);
        uint32_t a = (uint32_t)(centroid.x * 1024);
        uint32_t b = (uint32_t)(centroid.y * 1024);
        uint32_t c = (uint32_t)(centroid.z * 1024);
        return spaceBy3(a) | (spaceBy3(b) << 1) | (spaceBy3(c) << 2);
    }
    __host__ __device__ BBox startValue() const override {
        return BBox();
    }
    __host__ __device__ BBox elementToValue(Primitive * const &x) const override {
        return x->get_bbox();
    }
    __host__ __device__ void update(BBox &dst, const BBox &src) const override {
        dst.expand(src);
    }
    __device__ void atomicUpdate(BBox &dst, const BBox &src) const override {
        atomicMin(&(dst.min.x), src.min.x);
        atomicMin(&(dst.min.y), src.min.y);
        atomicMin(&(dst.min.z), src.min.z);
        atomicMax(&(dst.max.x), src.max.x);
        atomicMax(&(dst.max.y), src.max.y);
        atomicMax(&(dst.max.z), src.max.z);
    }
private:
    Vector3D min, max;
};


int main() {
    RadixTreeWrapper<BVHTree> bvh_wrapper(10);
    return 0;
}
