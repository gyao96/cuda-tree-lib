#include "pathtracer.h"

#include "scene/light.h"
#include "scene/sphere.h"
#include "scene/triangle.h"
#include <cmath>


using namespace CGL::SceneObjects;

namespace CGL {

PathTracer::PathTracer() {
  gridSampler = new UniformGridSampler2D();
  hemisphereSampler = new UniformHemisphereSampler3D();

  tm_gamma = 2.2f;
  tm_level = 1.0f;
  tm_key = 0.18;
  tm_wht = 5.0f;
}

PathTracer::~PathTracer() {
  delete gridSampler;
  delete hemisphereSampler;
}

void PathTracer::set_frame_size(size_t width, size_t height) {
  sampleBuffer.resize(width, height);
  sampleCountBuffer.resize(width * height);
}

void PathTracer::clear() {
  bvh = NULL;
  scene = NULL;
  camera = NULL;
  sampleBuffer.clear();
  sampleCountBuffer.clear();
  sampleBuffer.resize(0, 0);
  sampleCountBuffer.resize(0, 0);
}

void PathTracer::write_to_framebuffer(ImageBuffer &framebuffer, size_t x0,
                                      size_t y0, size_t x1, size_t y1) {
  sampleBuffer.toColor(framebuffer, x0, y0, x1, y1);
}

Spectrum
PathTracer::estimate_direct_lighting_hemisphere(const Ray &r,
                                                const Intersection &isect) {
  // Estimate the lighting from this intersection coming directly from a light.
  // For this function, sample uniformly in a hemisphere.

  // make a coordinate system for a hit point
  // with N aligned with the Z direction.
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  // w_out points towards the source of the ray (e.g.,
  // toward the camera if this is a primary ray)
  const Vector3D &hit_p = r.o + r.d * isect.t;
  const Vector3D &w_out = w2o * (-r.d);

  // This is the same number of total samples as
  // estimate_direct_lighting_importance (outside of delta lights). We keep the
  // same number of samples for clarity of comparison.
  int num_samples = scene->lights.size() * ns_area_light;
  Spectrum L_out;

  // TODO (Part 3): Write your sampling loop here
  // TODO BEFORE YOU BEGIN
  // UPDATE `est_radiance_global_illumination` to return direct lighting instead of normal shading
  Spectrum res;
  for (int i = 0; i < num_samples; ++i) {
    Vector3D w_in = hemisphereSampler->get_sample();
    Vector3D d_out = o2w * w_in;
    Ray ro(hit_p + EPS_D * d_out, d_out);
    Intersection isect_l;
    if (bvh->intersect(ro, &isect_l)) {
      res += zero_bounce_radiance(ro, isect_l) * isect.bsdf->f(w_out, w_in) * w_in.z * 2 * PI;
    }
  }

  return res / num_samples;
}

Spectrum
PathTracer::estimate_direct_lighting_importance(const Ray &r,
                                                const Intersection &isect) {
  // Estimate the lighting from this intersection coming directly from a light.
  // To implement importance sampling, sample only from lights, not uniformly in
  // a hemisphere.

  // make a coordinate system for a hit point
  // with N aligned with the Z direction.
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  // w_out points towards the source of the ray (e.g.,
  // toward the camera if this is a primary ray)
  const Vector3D &hit_p = r.o + r.d * isect.t;
  const Vector3D &w_out = w2o * (-r.d);
  
  Spectrum res;
  for (auto p = scene->lights.begin(); p != scene->lights.end(); p++) {
    int ns = (*p)->is_delta_light() ? 1 : ns_area_light;
    Spectrum res_l;
    for (int i = 0; i < ns; ++i) {
      Vector3D d_out;
      float dist, pdf;
      Spectrum emission = (*p)->sample_L(hit_p, &d_out, &dist, &pdf);
      if (dot(d_out, isect.n) < 0) continue;
      Vector3D w_in = w2o * d_out;
      Ray ro(hit_p + EPS_D * d_out, d_out);
      Intersection isect_l;
      if (!bvh->intersect(ro, &isect_l) || fabs(isect_l.t - dist) < 1e-5) {
        res_l += emission * isect.bsdf->f(w_out, w_in) * w_in.z / pdf;
      }
    }
    res += res_l / ns;
  }

  return res;
}

Spectrum PathTracer::zero_bounce_radiance(const Ray &r,
                                          const Intersection &isect) {
  // TODO: Part 3, Task 2
  // Returns the light that results from no bounces of light

  return isect.bsdf->get_emission();
}

Spectrum PathTracer::one_bounce_radiance(const Ray &r,
                                         const Intersection &isect) {
  // TODO: Part 3, Task 3
  // Returns either the direct illumination by hemisphere or importance sampling
  // depending on `direct_hemisphere_sample`

  // return estimate_direct_lighting_hemisphere(r, isect);
  return estimate_direct_lighting_importance(r, isect);
}

Spectrum PathTracer::at_least_one_bounce_radiance(const Ray &r,
                                                  const Intersection &isect) {
  const float prob_rr = 1;
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  Vector3D hit_p = r.o + r.d * isect.t;
  Vector3D w_out = w2o * (-r.d);

  Spectrum L_out;
  if (!isect.bsdf->is_delta())
    L_out += one_bounce_radiance(r, isect);

  Vector3D w_in;
  float pdf;
  Spectrum f = isect.bsdf->sample_f(w_out, &w_in, &pdf);
  Vector3D d_out = o2w * w_in;
  Ray ro(hit_p + EPS_D * d_out, d_out);
  ro.depth = r.depth + 1;

  Intersection isect_n;
  if (bvh->intersect(ro, &isect_n)) {
    Spectrum L;
    if (isect.bsdf->is_delta())
      L += zero_bounce_radiance(ro, isect_n);
    if (ro.depth < max_ray_depth && coin_flip(prob_rr))
      L += at_least_one_bounce_radiance(ro, isect_n) / prob_rr;
    L_out += L * f * abs_cos_theta(w_in) / pdf;
  }

  return L_out;
}

Spectrum PathTracer::est_radiance_global_illumination(const Ray &r) {
  Intersection isect;
  Spectrum L_out;

  // You will extend this in assignment 3-2.
  // If no intersection occurs, we simply return black.
  // This changes if you implement hemispherical lighting for extra credit.

  if (!bvh->intersect(r, &isect))
    return envLight ? envLight->sample_dir(r) : L_out;

  // The following line of code returns a debug color depending
  // on whether ray intersection with triangles or spheres has
  // been implemented.

  // REMOVE THIS LINE when you are ready to begin Part 3.
  // L_out = (isect.t == INF_D) ? debug_shading(r.d) : normal_shading(isect.n);

  // TODO (Part 3): Return the direct illumination.
  // return zero_bounce_radiance(r, isect) + one_bounce_radiance(r, isect);

  // TODO (Part 4): Accumulate the "direct" and "indirect"
  // parts of global illumination into L_out rather than just direct
  if (max_ray_depth == 0)
    return zero_bounce_radiance(r, isect);
  else
    return zero_bounce_radiance(r, isect) + at_least_one_bounce_radiance(r, isect);
}

void PathTracer::raytrace_pixel(size_t x, size_t y) {

  // TODO (Part 1.1):
  // Make a loop that generates num_samples camera rays and traces them
  // through the scene. Return the average Spectrum.
  // You should call est_radiance_global_illumination in this function.

  // TODO (Part 5):
  // Modify your implementation to include adaptive sampling.
  // Use the command line parameters "samplesPerBatch" and "maxTolerance"

  Vector2D origin = Vector2D(x, y); // bottom left corner of the pixel
  UniformGridSampler2D sampler;
  Spectrum res;
  
  /*
  int num_samples = ns_aa;          // total samples to evaluate
  for (int i = 0; i < num_samples; ++i) {
    Vector2D p = origin + sampler.get_sample();
    Ray ray = camera->generate_ray(p.x / sampleBuffer.w, p.y / sampleBuffer.h);
    res += est_radiance_global_illumination(ray);
  }
  */
  
  int num_samples = 0;
  double s1 = 0, s2 = 0;
  while (num_samples < ns_aa) {
    Vector2D p = origin + sampler.get_sample();
    // Ray ray = camera->generate_ray(p.x / sampleBuffer.w, p.y / sampleBuffer.h);
    Vector2D samplesForLens = gridSampler->get_sample();
    Ray ray = camera->generate_ray_for_thin_lens(p.x / sampleBuffer.w, p.y / sampleBuffer.h, samplesForLens.x, samplesForLens.y * 2 * PI);
    
    Spectrum res_this = est_radiance_global_illumination(ray);
    s1 += res_this.illum();
    s2 += res_this.illum() * res_this.illum();
    res += res_this;
    ++num_samples;
    if (num_samples % samplesPerBatch == 0) {
      double miu = s1 / num_samples;
      double sigma = sqrt(1.0 / (num_samples - 1) * (s2 - s1 * s1 / num_samples));
      if (1.96 * sigma / sqrt(num_samples) <= maxTolerance * miu) break;
    }
  }

  sampleBuffer.update_pixel(res / num_samples, x, y);
  sampleCountBuffer[x + y * sampleBuffer.w] = num_samples;
}

void PathTracer::autofocus(Vector2D loc) {
  Ray r = camera->generate_ray(loc.x / sampleBuffer.w, loc.y / sampleBuffer.h);
  Intersection isect;
  bvh->intersect(r, &isect);
  camera->focalDistance = isect.t;
}

} // namespace CGL
