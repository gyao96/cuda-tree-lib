#include "bbox.h"

#include "GL/glew.h"

#include <algorithm>
#include <iostream>

namespace CGL {

bool BBox::intersect(const Ray& r, double& t0, double& t1) const {

  // TODO (Part 2.2):
  // Implement ray - bounding box intersection test
  // If the ray intersected the bouding box within the range given by
  // t0, t1, update t0 and t1 with the new intersection times.
    double tminx = (min.x - r.o.x) / r.d.x;
    double tmaxx = (max.x - r.o.x) / r.d.x;
    if (tmaxx < tminx) std::swap(tmaxx, tminx);
    double tminy = (min.y - r.o.y) / r.d.y;
    double tmaxy = (max.y - r.o.y) / r.d.y;
    if (tmaxy < tminy) std::swap(tmaxy, tminy);
    double tminz = (min.z - r.o.z) / r.d.z;
    double tmaxz = (max.z - r.o.z) / r.d.z;
    if (tmaxz < tminz) std::swap(tmaxz, tminz);
    double tmin = std::max(tminx, tminy);
    double tmax = std::min(tmaxx, tmaxy);
    if (tminx > tmaxy || tminy > tmaxx) return false;
    if (tmin > tmaxz || tminz > tmax) return false;
    if (tminz > tmin) tmin = tminz;
    if (tmaxz < tmax) tmax = tmaxz;

    if (tmin < 0 && tmax > 0) return true;

    if (tmin > 0 || tmax > 0)
    {
        t0 = tmin > 0 ? tmin : 0;
        t1 = tmax > 0 ? tmax : 0;
        return tmin > r.min_t || tmax < r.max_t;
    }
    return false;
}

void BBox::draw(Color c, float alpha) const {

  glColor4f(c.r, c.g, c.b, alpha);

  // top
  glBegin(GL_LINE_STRIP);
  glVertex3d(max.x, max.y, max.z);
  glVertex3d(max.x, max.y, min.z);
  glVertex3d(min.x, max.y, min.z);
  glVertex3d(min.x, max.y, max.z);
  glVertex3d(max.x, max.y, max.z);
  glEnd();

  // bottom
  glBegin(GL_LINE_STRIP);
  glVertex3d(min.x, min.y, min.z);
  glVertex3d(min.x, min.y, max.z);
  glVertex3d(max.x, min.y, max.z);
  glVertex3d(max.x, min.y, min.z);
  glVertex3d(min.x, min.y, min.z);
  glEnd();

  // side
  glBegin(GL_LINES);
  glVertex3d(max.x, max.y, max.z);
  glVertex3d(max.x, min.y, max.z);
  glVertex3d(max.x, max.y, min.z);
  glVertex3d(max.x, min.y, min.z);
  glVertex3d(min.x, max.y, min.z);
  glVertex3d(min.x, min.y, min.z);
  glVertex3d(min.x, max.y, max.z);
  glVertex3d(min.x, min.y, max.z);
  glEnd();

}

std::ostream& operator<<(std::ostream& os, const BBox& b) {
  return os << "BBOX(" << b.min << ", " << b.max << ")";
}

} // namespace CGL
