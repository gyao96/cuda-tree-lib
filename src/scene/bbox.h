#ifndef CGL_BBOX_H
#define CGL_BBOX_H

#include <utility>
#include <algorithm>

#include "CGL/CGL.h"

#include "pathtracer/ray.h"
#include "util/cuda.h"

namespace CGL {

/**
 * Axis-aligned bounding box.
 * An AABB is given by two positions in space, the min and the max. An addition
 * component, the extent of the bounding box is stored as it is useful in a lot
 * of the operations on bounding boxes.
 */
struct BBox {

  Vector3D max;	    ///< min corner of the bounding box
  Vector3D min;	    ///< max corner of the bounding box
  Vector3D extent;  ///< extent of the bounding box (min -> max)

  /**
   * Constructor.
   * The default constructor creates a new bounding box which contains no
   * points.
   */
  __QUALIFIER__ BBox() {
    max = Vector3D(-INF_D, -INF_D, -INF_D);
    min = Vector3D( INF_D,  INF_D,  INF_D);
    extent = max - min;
  }

  /**
   * Constructor.
   * Creates a bounding box that includes a single point.
   */
  __QUALIFIER__ BBox(const Vector3D& p) : min(p), max(p) { extent = max - min; }

  /**
   * Constructor.
   * Creates a bounding box with given bounds.
   * \param min the min corner
   * \param max the max corner
   */
  __QUALIFIER__ BBox(const Vector3D& min, const Vector3D& max) :
       min(min), max(max) { extent = max - min; }

  /**
   * Constructor.
   * Creates a bounding box with given bounds (component wise).
   */
  __QUALIFIER__ BBox(const double minX, const double minY, const double minZ,
       const double maxX, const double maxY, const double maxZ) {
    min = Vector3D(minX, minY, minZ);
    max = Vector3D(maxX, maxY, maxZ);
		extent = max - min;
  }

  /**
   * Expand the bounding box to include another (union).
   * If the given bounding box is contained within *this*, nothing happens.
   * Otherwise *this* is expanded to the minimum volume that contains the
   * given input.
   * \param bbox the bounding box to be included
   */
  __QUALIFIER__ void expand(const BBox& bbox) {
    min.x = fmin(min.x, bbox.min.x);
    min.y = fmin(min.y, bbox.min.y);
    min.z = fmin(min.z, bbox.min.z);
    max.x = fmax(max.x, bbox.max.x);
    max.y = fmax(max.y, bbox.max.y);
    max.z = fmax(max.z, bbox.max.z);
    extent = max - min;
  }

  /**
   * Expand the bounding box to include a new point in space.
   * If the given point is already inside *this*, nothing happens.
   * Otherwise *this* is expanded to a minimum volume that contains the given
   * point.
   * \param p the point to be included
   */
  __QUALIFIER__ void expand(const Vector3D& p) {
    min.x = fmin(min.x, p.x);
    min.y = fmin(min.y, p.y);
    min.z = fmin(min.z, p.z);
    max.x = fmax(max.x, p.x);
    max.y = fmax(max.y, p.y);
    max.z = fmax(max.z, p.z);
    extent = max - min;
  }

  __QUALIFIER__ Vector3D centroid() const {
    return (min + max) / 2;
  }

  /**
   * Compute the surface area of the bounding box.
   * \return surface area of the bounding box.
   */
  __QUALIFIER__ double surface_area() const {
    if (empty()) return 0.0;
    return 2 * (extent.x * extent.z +
                extent.x * extent.y +
                extent.y * extent.z);
  }

  /**
   * Check if bounding box is empty.
   * Bounding box that has no size is considered empty. Note that since
   * bounding box are used for objects with positive volumes, a bounding
   * box of zero size (empty, or contains a single vertex) are considered
   * empty.
   */
  __QUALIFIER__ bool empty() const {
    return min.x > max.x || min.y > max.y || min.z > max.z;
  }

  /**
   * Ray - bbox intersection.
   * Intersects ray with bounding box, does not store shading information.
   * \param r the ray to intersect with
   * \param t0 lower bound of intersection time
   * \param t1 upper bound of intersection time
   */
  __QUALIFIER__ bool intersect(const Ray& r, double& t0, double& t1) const;

  /**
   * Draw box wireframe with OpenGL.
   * \param c color of the wireframe
   */
  void draw(Color c, float) const;
};

std::ostream& operator<<(std::ostream& os, const BBox& b);

} // namespace CGL

#endif // CGL_BBOX_H
