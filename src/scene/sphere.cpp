#include "sphere.h"

#include <cmath>

#include "pathtracer/bsdf.h"
#include "util/sphere_drawing.h"

namespace CGL {
namespace SceneObjects {

bool Sphere::test(const Ray &r, double &t1, double &t2) const {

  // TODO (Part 1.4):
  // Implement ray - sphere intersection test.
  // Return true if there are intersections and writing the
  // smaller of the two intersection times in t1 and the larger in t2.

  return true;
}

bool Sphere::has_intersection(const Ray &r) const {

  // TODO (Part 1.4):
  // Implement ray - sphere intersection.
  // Note that you might want to use the the Sphere::test helper here.
  
  double a = dot(r.d, r.d);
  double b = 2 * dot(r.o - o, r.d);
  double c = dot(r.o - o, r.o - o) - r2;
  if (b * b < 4 * a * c) return false;
  
  double t1 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
  double t2 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
  double t = t1 >= r.min_t ? t1 : t2;
  if (t < r.min_t || t > r.max_t) return false;
  
  r.max_t = t;
  return true;
}

bool Sphere::intersect(const Ray &r, Intersection *i) const {

  // TODO (Part 1.4):
  // Implement ray - sphere intersection.
  // Note again that you might want to use the the Sphere::test helper here.
  // When an intersection takes place, the Intersection data should be updated
  // correspondingly.
  
  double a = dot(r.d, r.d);
  double b = 2 * dot(r.o - o, r.d);
  double c = dot(r.o - o, r.o - o) - r2;
  if (b * b < 4 * a * c) return false;
  
  double t1 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
  double t2 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
  double t = t1 >= r.min_t ? t1 : t2;
  if (t < r.min_t || t > r.max_t) return false;
  
  r.max_t = t;
  i->t = t;
  i->primitive = this;
  i->bsdf = get_bsdf();
  i->n = r.o + t * r.d - o;
  i->n.normalize();

  return true;
}

void Sphere::draw(const Color &c, float alpha) const {
  Misc::draw_sphere_opengl(o, r, c);
}

void Sphere::drawOutline(const Color &c, float alpha) const {
  // Misc::draw_sphere_opengl(o, r, c);
}

} // namespace SceneObjects
} // namespace CGL
