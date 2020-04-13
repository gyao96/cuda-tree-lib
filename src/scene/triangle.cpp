#include "triangle.h"

#include "CGL/CGL.h"
#include "GL/glew.h"

namespace CGL {
namespace SceneObjects {

// Helpers
double point_line_distance(Vector3D p, Vector3D vtx1, Vector3D vtx2) {
  return cross(vtx2 - vtx1, p - vtx1).norm() / (vtx2 - vtx1).norm();
}
template <typename T>
T barycentric(Vector3D p, Vector3D vtx1, Vector3D vtx2, Vector3D vtx3, T t1, T t2, T t3) {
  double alpha = point_line_distance(p, vtx2, vtx3) / point_line_distance(vtx1, vtx2, vtx3);
  double beta = point_line_distance(p, vtx3, vtx1) / point_line_distance(vtx2, vtx3, vtx1);
  double gamma = 1 - alpha - beta;
  return t1 * alpha + t2 * beta + t3 * gamma;
}

Triangle::Triangle(const Mesh *mesh, size_t v1, size_t v2, size_t v3) {
  p1 = mesh->positions[v1];
  p2 = mesh->positions[v2];
  p3 = mesh->positions[v3];
  n1 = mesh->normals[v1];
  n2 = mesh->normals[v2];
  n3 = mesh->normals[v3];
  bbox = BBox(p1);
  bbox.expand(p2);
  bbox.expand(p3);

  bsdf = mesh->get_bsdf();
}

BBox Triangle::get_bbox() const { return bbox; }

bool Triangle::has_intersection(const Ray &r) const {
  // Part 1, Task 3: implement ray-triangle intersection
  // The difference between this function and the next function is that the next
  // function records the "intersection" while this function only tests whether
  // there is a intersection.
  
  Vector3D normal = cross(p2 - p1, p3 - p2);
  normal.normalize();
  double t = dot(p1 - r.o, normal) / dot(r.d, normal);
  if (t < r.min_t || t > r.max_t) return false;

  Vector3D isectp = r.o + t * r.d;
  if (dot(cross(p2 - p1, isectp - p1), normal) < 0 ||
      dot(cross(p3 - p2, isectp - p2), normal) < 0 ||
      dot(cross(p1 - p3, isectp - p3), normal) < 0) return false;
  
  r.max_t = t;
  return true;
}

bool Triangle::intersect(const Ray &r, Intersection *isect) const {
  // Part 1, Task 3:
  // implement ray-triangle intersection. When an intersection takes
  // place, the Intersection data should be updated accordingly

  Vector3D normal = cross(p2 - p1, p3 - p2);
  normal.normalize();
  double t = dot(p1 - r.o, normal) / dot(r.d, normal);
  if (t < r.min_t || t > r.max_t) return false;

  Vector3D isectp = r.o + t * r.d;
  if (dot(cross(p2 - p1, isectp - p1), normal) < 0 ||
      dot(cross(p3 - p2, isectp - p2), normal) < 0 ||
      dot(cross(p1 - p3, isectp - p3), normal) < 0) return false;

  r.max_t = t;
  isect->t = t;
  isect->primitive = this;
  isect->bsdf = get_bsdf();
  isect->n = barycentric(isectp, p1, p2, p3, n1, n2, n3);

  return true;
}

void Triangle::draw(const Color &c, float alpha) const {
  glColor4f(c.r, c.g, c.b, alpha);
  glBegin(GL_TRIANGLES);
  glVertex3d(p1.x, p1.y, p1.z);
  glVertex3d(p2.x, p2.y, p2.z);
  glVertex3d(p3.x, p3.y, p3.z);
  glEnd();
}

void Triangle::drawOutline(const Color &c, float alpha) const {
  glColor4f(c.r, c.g, c.b, alpha);
  glBegin(GL_LINE_LOOP);
  glVertex3d(p1.x, p1.y, p1.z);
  glVertex3d(p2.x, p2.y, p2.z);
  glVertex3d(p3.x, p3.y, p3.z);
  glEnd();
}

} // namespace SceneObjects
} // namespace CGL
