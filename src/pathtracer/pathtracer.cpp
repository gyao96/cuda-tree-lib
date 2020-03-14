#include "pathtracer.h"

#include "scene/light.h"
#include "scene/sphere.h"
#include "scene/triangle.h"


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
  for (size_t i = 0; i < num_samples; i++)
  {
      Vector3D w_in = hemisphereSampler->get_sample();  // in local space, as is w_out
      Vector3D w_in_global = o2w * w_in;
      Ray ri(hit_p + EPS_D * w_in_global, w_in_global);
      Intersection hit_light;
      if (bvh->intersect(ri, &hit_light))
      {
          double pdf = 1.0 / (2 * PI);
          double costheta = w_in.z; // cos(theta) = (0,0,1) dot w_in
          L_out += hit_light.bsdf->get_emission() * isect.bsdf->f(w_out, w_in) * costheta / pdf;
      }
  }

  return L_out / num_samples;
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
  Spectrum L_out;

    Vector3D w_in_global;
    float distToLight;
    float pdf;
    for (const SceneLight *light: scene->lights)
    {
        if (light->is_delta_light())
        {
            Spectrum pRadiance = light->sample_L(hit_p, &w_in_global, &distToLight, &pdf);
            Ray ri(hit_p + EPS_D * w_in_global, w_in_global);
            ri.max_t = distToLight;
            Intersection shadow_ray;
            if (dot(isect.n, w_in_global) > -EPS_D && (!bvh->has_intersection(ri)))
            {
                Vector3D w_in = w2o * w_in_global;
                double costheta = w_in.z;
                L_out += pRadiance * isect.bsdf->f(w_out, w_in) * costheta / pdf;
            }
        }
        else
        {
            Spectrum L_area;
            for (size_t i = 0; i < ns_area_light; i++)
            {
                Spectrum pRadiance = light->sample_L(hit_p, &w_in_global, &distToLight, &pdf);
                Ray ri(hit_p + EPS_D * w_in_global, w_in_global);
                ri.max_t = distToLight;
                Intersection shadow_ray;
                if (dot(isect.n, w_in_global) > -EPS_D && (!bvh->has_intersection(ri)))
                {
                    Vector3D w_in = w2o * w_in_global;
                    double costheta = w_in.z;
                    L_area += pRadiance * isect.bsdf->f(w_out, w_in) * costheta / pdf;
                }
            }
            L_out += (L_area / ns_area_light);
        }
    }

  return L_out;
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

    if (direct_hemisphere_sample)
        return estimate_direct_lighting_hemisphere(r, isect);
    else
        return estimate_direct_lighting_importance(r, isect);

}

Spectrum PathTracer::at_least_one_bounce_radiance(const Ray &r,
                                                  const Intersection &isect) {
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  Vector3D hit_p = r.o + r.d * isect.t;
  Vector3D w_out = w2o * (-r.d);

  Spectrum L_out;
  //if (r.depth < max_ray_depth)
      L_out += one_bounce_radiance(r, isect);
  Vector3D w_in;
  float pdf;
  Spectrum irr = isect.bsdf->sample_f(w_out, &w_in, &pdf);
  Vector3D w_in_global = o2w * w_in;
  Intersection next_isect;
  Ray s_ray(hit_p + EPS_D * w_in_global, w_in_global);
  s_ray.depth = r.depth - 1;
  bool hit = bvh->intersect(s_ray, &next_isect);
  double cpdf = 0.7;
  if (r.depth > 0 && hit && coin_flip(cpdf))
  {
      L_out += at_least_one_bounce_radiance(s_ray, next_isect) * irr * cos_theta(w_in) / pdf / cpdf;
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
    return L_out;

  // The following line of code returns a debug color depending
  // on whether ray intersection with triangles or spheres has
  // been implemented.

  // REMOVE THIS LINE when you are ready to begin Part 3.
  // L_out = (isect.t == INF_D) ? debug_shading(r.d) : normal_shading(isect.n);

  // TODO (Part 3): Return the direct illumination.
  L_out += zero_bounce_radiance(r, isect);
  // TODO (Part 4): Accumulate the "direct" and "indirect"
  // L_out += one_bounce_radiance(r, isect);
  // parts of global illumination into L_out rather than just direct
  L_out += at_least_one_bounce_radiance(r, isect);
  return L_out;
}

void PathTracer::raytrace_pixel(size_t x, size_t y) {

  // TODO (Part 1.1):
  // Make a loop that generates num_samples camera rays and traces them
  // through the scene. Return the average Spectrum.
  // You should call est_radiance_global_illumination in this function.

  // TODO (Part 5):
  // Modify your implementation to include adaptive sampling.
  // Use the command line parameters "samplesPerBatch" and "maxTolerance"

  int num_samples = ns_aa;          // total samples to evaluate
  Vector2D origin = Vector2D(x, y); // bottom left corner of the pixel

  Vector2D norm_origin(origin.x / sampleBuffer.w, origin.y / sampleBuffer.h);
  Spectrum avg_spectrum(0.0);
  float s1 = 0., s2 = 0.;
  size_t i;
  for (i = 0; i < num_samples; i++)
  {
      Vector2D sample = gridSampler->get_sample();
      double normx = double(x + sample.x) / sampleBuffer.w; 
      double normy = double(y + sample.y) / sampleBuffer.h;
      Ray sample_ray = camera->generate_ray(normx, normy);
      sample_ray.depth = max_ray_depth;
      Spectrum ill = est_radiance_global_illumination(sample_ray);
      avg_spectrum += ill;
      s1 += ill.illum();
      s2 += ill.illum() * ill.illum();
      if ((i+1) % samplesPerBatch == 0)
      {
          float miu = s1 / (i + 1);
          float sigma2 = (s2 - s1 * s1 / (i + 1)) / i;
          if (1.96 * sqrt(sigma2 / (i + 1)) <= maxTolerance * miu)
          {
              break;
          }
      }
  }
  num_samples = i + 1;
  avg_spectrum /= num_samples;
  sampleBuffer.update_pixel(avg_spectrum, x, y);
  sampleCountBuffer[x + y * sampleBuffer.w] = num_samples;
}

} // namespace CGL