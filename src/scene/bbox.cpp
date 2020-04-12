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
  
  double tmin = t0, tmax = t1, ti0, ti1;
  ti0 = (min.x - r.o.x) / r.d.x;
  ti1 = (max.x - r.o.x) / r.d.x;
  tmin = std::max(tmin, std::min(ti0, ti1));
  tmax = std::min(tmax, std::max(ti0, ti1));
  ti0 = (min.y - r.o.y) / r.d.y;
  ti1 = (max.y - r.o.y) / r.d.y;
  tmin = std::max(tmin, std::min(ti0, ti1));
  tmax = std::min(tmax, std::max(ti0, ti1));
  ti0 = (min.z - r.o.z) / r.d.z;
  ti1 = (max.z - r.o.z) / r.d.z;
  tmin = std::max(tmin, std::min(ti0, ti1));
  tmax = std::min(tmax, std::max(ti0, ti1));
  
  if (tmin > tmax) return false;
  t0 = tmin;
  t1 = tmax;
  return true;
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
