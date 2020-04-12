#include "camera.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include "CGL/misc.h"
#include "CGL/vector2D.h"
#include "CGL/vector3D.h"

using std::cout;
using std::endl;
using std::max;
using std::min;
using std::ifstream;
using std::ofstream;

namespace CGL {

using Collada::CameraInfo;

Ray Camera::generate_ray_for_thin_lens(double x, double y, double rndR, double rndTheta) const {
  // Part 2, Task 4:
  // compute position and direction of ray from the input sensor sample coordinate.
  // Note: use rndR and rndTheta to uniformly sample a unit disk.

  double sensorX = (x - 0.5) * 2 * tan(hFov / 2 / 180 * PI);
  double sensorY = (y - 0.5) * 2 * tan(vFov / 2 / 180 * PI);
  Vector3D pFocus = Vector3D(sensorX, sensorY, -1) * focalDistance;
  Vector3D pLens(lensRadius * sqrt(rndR) * cos(rndTheta), lensRadius * sqrt(rndR) * sin(rndTheta), 0);
  Vector3D direction = (pFocus - pLens).unit();
  Ray ray(pos + pLens, c2w * direction);
  ray.min_t = nClip;
  ray.max_t = fClip;
  return ray;
}


} // namespace CGL
