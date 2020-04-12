#include "bvh.h"

#include "CGL/CGL.h"
#include "triangle.h"

#include <iostream>
#include <stack>

using namespace std;

namespace CGL {
namespace SceneObjects {

BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t max_leaf_size) {

  primitives = std::vector<Primitive *>(_primitives);
  root = construct_bvh(primitives.begin(), primitives.end(), max_leaf_size);
}

BVHAccel::~BVHAccel() {
  if (root)
    delete root;
}

BBox BVHAccel::get_bbox() const { return root->bb; }

void BVHAccel::draw(BVHNode *node, const Color &c, float alpha) const {
  if (node->isLeaf()) {
    for (auto p = node->start; p != node->end; p++) {
      (*p)->draw(c, alpha);
    }
  } else {
    draw(node->l, c, alpha);
    draw(node->r, c, alpha);
  }
}

void BVHAccel::drawOutline(BVHNode *node, const Color &c, float alpha) const {
  if (node->isLeaf()) {
    for (auto p = node->start; p != node->end; p++) {
      (*p)->drawOutline(c, alpha);
    }
  } else {
    drawOutline(node->l, c, alpha);
    drawOutline(node->r, c, alpha);
  }
}

std::vector<std::vector<Primitive *>> primitives_node;
BVHNode *BVHAccel::construct_bvh(std::vector<Primitive *>::iterator start,
                                 std::vector<Primitive *>::iterator end,
                                 size_t max_leaf_size) {

  // TODO (Part 2.1):
  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.

  BBox bbox;

  int cnt = 0;
  for (auto p = start; p != end; p++) {
    ++cnt;
    BBox bb = (*p)->get_bbox();
    bbox.expand(bb);
  }
  BVHNode *node = new BVHNode(bbox);

  if (cnt <= max_leaf_size) {
    primitives_node.push_back(std::vector<Primitive *>());
    for (auto p = start; p != end; p++) {
      primitives_node.back().push_back(*p);
    }
    node->start = primitives_node.back().begin();
    node->end = primitives_node.back().end();
    node->l = node->r = NULL;
    return node;
  }
  
  const int X = 0, Y = 1, Z = 2;
  int axis = 0;
  if (bbox.extent.x >= bbox.extent.y && bbox.extent.x >= bbox.extent.z) axis = X;
  else if (bbox.extent.y >= bbox.extent.z && bbox.extent.y >= bbox.extent.x) axis = Y;
  else if (bbox.extent.z >= bbox.extent.x && bbox.extent.z >= bbox.extent.y) axis = Z;
  else throw;
  
  double centroid = 0;
  for (auto p = start; p != end; p++) {
    if (axis == X) centroid += (*p)->get_bbox().centroid().x;
    else if (axis == Y) centroid += (*p)->get_bbox().centroid().y;
    else if (axis == Z) centroid += (*p)->get_bbox().centroid().z;
    else throw;
  }
  centroid /= cnt;

  std::vector<Primitive *> left, right;
  for (auto p = start; p != end; p++) {
    double pivot = 0;
    if (axis == X) pivot = (*p)->get_bbox().centroid().x;
    else if (axis == Y) pivot = (*p)->get_bbox().centroid().y;
    else if (axis == Z) pivot = (*p)->get_bbox().centroid().z;
    else throw;
    if (pivot <= centroid) left.push_back(*p);
    else right.push_back(*p);
  }

  if (left.empty() || right.empty()) {
    left.clear();
    right.clear();
    int i = 0;
    auto p = start;
    for (; i < cnt / 2; ++i, ++p) left.push_back(*p);
    for (; p != end; ++p) right.push_back(*p);
  }
  node->l = construct_bvh(left.begin(), left.end(), max_leaf_size);
  node->r = construct_bvh(right.begin(), right.end(), max_leaf_size);
  return node;
}

bool BVHAccel::has_intersection(const Ray &ray, BVHNode *node) const {
  // TODO (Part 2.3):
  // Fill in the intersect function.
  // Take note that this function has a short-circuit that the
  // Intersection version cannot, since it returns as soon as it finds
  // a hit, it doesn't actually have to find the closest hit.

  double t0 = ray.min_t, t1 = ray.max_t;
  if (!node->bb.intersect(ray, t0, t1)) return false;
  Ray ray_ = ray;
  ray_.min_t = t0;
  ray_.max_t = t1;
  
  if (node->isLeaf()) {
    for (auto p = node->start; p != node->end; ++p) {
      if ((*p)->has_intersection(ray_)) return true;
    }
    return false;
  }
  
  bool hit = has_intersection(ray_, node->l) || has_intersection(ray_, node->r);
  return hit;
}

bool BVHAccel::intersect(const Ray &ray, Intersection *i, BVHNode *node) const {
  // TODO (Part 2.3):
  // Fill in the intersect function.

  double t0 = ray.min_t, t1 = ray.max_t;
  if (!node->bb.intersect(ray, t0, t1)) return false;
  Ray ray_ = ray;
  ray_.min_t = t0;
  ray_.max_t = t1;

  if (node->isLeaf()) {
    bool hit = false;
    for (auto p = node->start; p != node->end; ++p) {
      Intersection isect;
      if ((*p)->intersect(ray_, &isect)) {
        if (!hit || isect.t < i->t) {
          *i = isect;
        }
        hit = true;
      }
    }
    return hit;
  }
  
  Intersection isect1, isect2;
  bool hit1 = intersect(ray_, &isect1, node->l);
  bool hit2 = intersect(ray_, &isect2, node->r);
  if (hit1 && (!hit2 || isect1.t <= isect2.t)) *i = isect1;
  else if (hit2 && (!hit1 || isect2.t <= isect1.t)) *i = isect2;
  if (hit1 || hit2) {
    ray.max_t = i->t;
  }
  return hit1 || hit2;
}

} // namespace SceneObjects
} // namespace CGL
