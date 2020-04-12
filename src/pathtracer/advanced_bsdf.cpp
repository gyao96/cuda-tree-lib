#include "bsdf.h"

#include <algorithm>
#include <iostream>
#include <utility>


using std::max;
using std::min;
using std::swap;

namespace CGL {

// Mirror BSDF //

Spectrum MirrorBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum MirrorBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  // TODO:
  // Implement MirrorBSDF
  reflect(wo, wi);
  *pdf = 1;
  return reflectance / abs_cos_theta(*wi);
}

// Microfacet BSDF //

double MicrofacetBSDF::G(const Vector3D& wo, const Vector3D& wi) {
  return 1.0 / (1.0 + Lambda(wi) + Lambda(wo));
}

double MicrofacetBSDF::D(const Vector3D& h) {
  // TODO: proj3-2, part 3
  // Compute Beckmann normal distribution function (NDF) here.
  // You will need the roughness alpha.
  double tan2_thetah = (h.x * h.x + h.y * h.y) / (h.z * h.z);
  double cos2_theta = h.z * h.z;
  return std::exp(-tan2_thetah / (alpha * alpha)) / (PI * alpha * alpha * cos2_theta * cos2_theta);
}

Spectrum MicrofacetBSDF::F(const Vector3D& wi) {
  // TODO: proj3-2, part 3
  // Compute Fresnel term for reflection on dielectric-conductor interface.
  // You will need both eta and etaK, both of which are Spectrum.
  Spectrum term1 = eta * eta + k * k;   // eta^2+k^2
  Spectrum term2 = 2 * eta * cos_theta(wi);   // 2*eta*cos(theta_i)
  Spectrum term3 = cos_theta(wi) * cos_theta(wi);   // cos^2(theta_i)
  Spectrum Rs = (term1 - term2 + term3) / (term1 + term2 + term3);
  Spectrum Rp = (term1 * term3 - term2 + 1) / (term1 * term3 + term2 + 1);
  return (Rs + Rp) / 2;
}

Spectrum MicrofacetBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  // TODO: proj3-2, part 3
  // Implement microfacet model here.
  if (wo.z <= 0 || wi.z <= 0) return Spectrum();
  return F(wi) * G(wo, wi) * D((wo + wi).unit()) / (4 * wo.z * wi.z);
}

Spectrum MicrofacetBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  // TODO: proj3-2, part 3
  // *Importance* sample Beckmann normal distribution function (NDF) here.
  // Note: You should fill in the sampled direction *wi and the corresponding *pdf,
  //       and return the sampled BRDF value.
  Vector2D r = sampler.get_sample();
  double r1 = r.x, r2 = r.y;
  double theta = atan(sqrt(-alpha * alpha * log(1 - r1)));
  double phi = 2 * PI * r2;
  Vector3D h(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
  *wi = h * dot(wo, h) * 2 - wo;
  
  double cos3theta = cos(theta); cos3theta = cos3theta * cos3theta * cos3theta;
  double tan2theta = tan(theta); tan2theta = tan2theta * tan2theta;
  double p_theta = 2 * sin(theta) / (alpha * alpha * cos3theta) * exp(-tan2theta / (alpha * alpha));
  double p_phi = 1 / (2 * PI);
  *pdf = (p_theta * p_phi) / sin(theta) / (4 * dot(*wi, h));
  // *wi = cosineHemisphereSampler.get_sample(pdf); //placeholder
  return f(wo, *wi);
}

// Refraction BSDF //

Spectrum RefractionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum RefractionBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  // TODO:
  // Implement RefractionBSDF
  return Spectrum();
}

// Glass BSDF //

Spectrum GlassBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum GlassBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  // TODO:
  // Compute Fresnel coefficient and use it as the probability of reflection
  // - Fundamentals of Computer Graphics page 305
  if (!refract(wo, wi, ior)) {
    reflect(wo, wi);
    *pdf = 1;
    return reflectance / abs_cos_theta(*wi);
  }
  else {
    float R0 = (ior - 1) / (ior + 1);
    R0 = R0 * R0;
    float R = R0 + (1 - R0) * std::pow(1 - abs_cos_theta(*wi), 5);
    if (coin_flip(R)) {
      reflect(wo, wi);
      *pdf = R;
      return R * reflectance / abs_cos_theta(*wi);
    }
    else {
      *pdf = 1 - R;
      float eta = (wo.z > 0 ? 1 / ior : ior);
      return (1 - R) * transmittance / abs_cos_theta(*wi) * (eta * eta);
    }
  }
}

void BSDF::reflect(const Vector3D& wo, Vector3D* wi) {
  // TODO:
  // Implement reflection of wo about normal (0,0,1) and store result in wi.
  *wi = Vector3D(-wo.x, -wo.y, wo.z);
}

bool BSDF::refract(const Vector3D& wo, Vector3D* wi, float ior) {
  // TODO:
  // Use Snell's Law to refract wo surface and store result ray in wi.
  // Return false if refraction does not occur due to total internal reflection
  // and true otherwise. When dot(wo,n) is positive, then wo corresponds to a
  // ray entering the surface through vacuum.
  if (wo.z > 0) {
    float eta = 1 / ior;
    wi->x = -eta * wo.x;
    wi->y = -eta * wo.y;
    wi->z = -sqrt(1 - eta * eta * (1 - wo.z * wo.z));
  }
  else {
    float eta = ior;
    if (eta * eta * (1 - wo.z * wo.z) > 1) return false;
    wi->x = -eta * wo.x;
    wi->y = -eta * wo.y;
    wi->z = sqrt(1 - eta * eta * (1 - wo.z * wo.z));
  }
  return true;
}

} // namespace CGL
