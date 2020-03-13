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
  primitives.clear();
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

BVHNode *BVHAccel::construct_bvh(std::vector<Primitive *>::iterator start,
                                 std::vector<Primitive *>::iterator end,
                                 size_t max_leaf_size, size_t level) {

  // TODO (Part 2.1):
  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.

  BBox bbox;
  int nodeEleCount = 0;
  Vector3D avgCentroid(0.0);
  for (auto p = start; p != end; p++) {
    BBox bb = (*p)->get_bbox();
    bbox.expand(bb);
    avgCentroid += bb.centroid();
    nodeEleCount++;
  }
  BVHNode *node = new BVHNode(bbox);
  if (nodeEleCount <= max_leaf_size)
  {
      node->start = start;
      node->end = end;

      return node;
  }
  
      avgCentroid /= nodeEleCount;
      int key = level % 3;
      std::vector<Primitive*>::iterator split_iter;
      if (key == 0)
      {
          split_iter = std::partition(start, end, [avgCentroid](const Primitive* ele)
              {
                  return ele->get_bbox().centroid().z < avgCentroid.z;
              });
      }
      else if (key == 1)
      {
          split_iter = std::partition(start, end, [avgCentroid](const Primitive* ele)
              {
                  return ele->get_bbox().centroid().y < avgCentroid.y;
              });
      }
      else if (key == 2)
      {
          split_iter = std::partition(start, end, [avgCentroid](const Primitive* ele)
              {
                  return ele->get_bbox().centroid().x < avgCentroid.x;
              });
      }
      else
      {
          perror("Exceed partition switch bound\n");
      }
      
      if (end - split_iter <= 0)
      {
          split_iter--;
      }
      else if (split_iter - start <= 0)
      {
          split_iter++;
      }
      assert(end - split_iter > 0 && split_iter - start > 0);
      node->l = construct_bvh(start, split_iter, max_leaf_size, level + 1);
      node->r = construct_bvh(split_iter, end, max_leaf_size, level + 1);
      return node;
}

bool BVHAccel::has_intersection(const Ray &ray, BVHNode *node) const {
  // TODO (Part 2.3):
  // Fill in the intersect function.
  // Take note that this function has a short-circuit that the
  // Intersection version cannot, since it returns as soon as it finds
  // a hit, it doesn't actually have to find the closest hit.
    double t0, t1;
    if (!node->bb.intersect(ray, t0, t1))
        return false;
    if (node->isLeaf())
    {
        for (auto it = node->start; it != node->end; it++)
        {
            if ((*it)->has_intersection(ray))
            {
                return true;
            }
        }
        return false;
    }
    bool hit1 = has_intersection(ray, node->l);
    bool hit2 = has_intersection(ray, node->r);
    return hit1 || hit2;

}

bool BVHAccel::intersect(const Ray &ray, Intersection *i, BVHNode *node) const {
  // TODO (Part 2.3):
  // Fill in the intersect function.
    double t0, t1;
    if (!node->bb.intersect(ray, t0, t1))
        return false;
    if (node->isLeaf())
    {
        bool hit = false;
        for (auto it = node->start; it != node->end; it++)
        {
            total_isects++;
            hit = (*it)->intersect(ray, i) || hit;
        }
        return hit;
    }
    bool hit1 = intersect(ray, i, node->l);
    bool hit2 = intersect(ray, i, node->r);
    return hit1 || hit2;
}

} // namespace SceneObjects
} // namespace CGL
