#ifndef CGL_VECTOR3D_H
#define CGL_VECTOR3D_H

#include "CGL.h"
#include "color.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>
#include <new>

#ifdef __AVX__
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

namespace CGL {

/**
 * Defines 3D vectors.
 */
class Vector3D {
public:

  // components
#ifdef __AVX__
  union {
    struct {
      double x, y, z;
    };
    struct {
      double r, g, b;
    };
    struct {
      alignas(16) __m128d __vec;
      double _z;
    };
  };
#else
  union {
    struct {
      double x, y, z;
    };
    struct {
      double r, g, b;
    };
  };
#endif

  /**
   * Constructor.
   * Initializes tp vector (0,0,0).
   */
  __QUALIFIER__ Vector3D() : x(0.0), y(0.0), z(0.0) { }

  /**
   * Constructor.
   * Initializes to vector (x,y,z).
   */
  __QUALIFIER__ Vector3D( double x, double y, double z) : x( x ), y( y ), z( z ) { }

  /**
   * Constructor.
   * Initializes to vector (c,c,c)
   */
  __QUALIFIER__ Vector3D( double c ) : x( c ), y( c ), z( c ) { }

  #ifdef __AVX__
  Vector3D( __m128d v, double z ) : __vec(v), _z(z) { }
  #endif

  /**
   * Constructor.
   * Initializes from existing vector
   */
  __QUALIFIER__ Vector3D( const Vector3D& v ) : x( v.x ), y( v.y ), z( v.z ) { }

  // returns reference to the specified component (0-based indexing: x, y, z)
  __QUALIFIER__ inline double& operator[] ( const int& index ) {
    return ( &x )[ index ];
  }

  // returns const reference to the specified component (0-based indexing: x, y, z)
  __QUALIFIER__ inline const double& operator[] ( const int& index ) const {
    return ( &x )[ index ];
  }

  __QUALIFIER__ inline bool operator==( const Vector3D& v) const {
    return v.x == x && v.y == y && v.z == z;
  }

  // negation
  __QUALIFIER__ inline Vector3D operator-( void ) const {
    return Vector3D( -x, -y, -z );
  }

  // addition
  __QUALIFIER__ inline Vector3D operator+( const Vector3D& v ) const {
#ifdef __AVX__
    return Vector3D(_mm_add_pd(__vec, v.__vec), _z + v._z);
#else
    return Vector3D(x + v.x, y + v.y, z + v.z);
#endif
  }

  // subtraction
  __QUALIFIER__ inline Vector3D operator-( const Vector3D& v ) const {
#ifdef __AVX__
    return Vector3D( _mm_sub_pd(__vec, v.__vec), _z - v._z );
#else
    return Vector3D( x - v.x, y - v.y, z - v.z );
#endif
  }

  // element wise multiplication
  __QUALIFIER__ inline Vector3D operator*(const Vector3D& v) const {
#ifdef __AVX__
    return Vector3D(_mm_mul_pd(__vec, v.__vec), _z * v._z);
#else
    return Vector3D(x * v.x, y * v.y, z * v.z);
#endif
  }
  
  // element wise division
  __QUALIFIER__ inline Vector3D operator/(const Vector3D& v) const {
#ifdef __AVX__
    return Vector3D(_mm_div_pd(__vec, v.__vec), _z / v._z);
#else
    return Vector3D(x / v.x, y / v.y, z / v.z);
#endif
  }

  // right scalar multiplication
  __QUALIFIER__ inline Vector3D operator*( const double& c ) const {
    return Vector3D( x * c, y * c, z * c );
  }

  // scalar division
  __QUALIFIER__ inline Vector3D operator/( const double& c ) const {
    const double rc = 1.0 / c;
    return Vector3D( rc * x, rc * y, rc * z );
  }

  // addition / assignment
  __QUALIFIER__ inline void operator+=( const Vector3D& v ) {
#ifdef __AVX__
    __vec = _mm_add_pd(__vec, v.__vec);
    _z += v._z;
#else
    x += v.x; y += v.y; z += v.z;
#endif
  }

  // subtraction / assignment
  __QUALIFIER__ inline void operator-=( const Vector3D& v ) {
#ifdef __AVX__
    __vec = _mm_sub_pd(__vec, v.__vec);
    _z -= v._z;
#else
    x -= v.x; y -= v.y; z -= v.z;
#endif
  }

  // scalar multiplication / assignment
  __QUALIFIER__ inline void operator*=( const double& c ) {
    x *= c; y *= c; z *= c;
  }

  // scalar division / assignment
  __QUALIFIER__ inline void operator/=( const double& c ) {
    (*this) *= ( 1./c );
  }

  /**
   * Returns per entry reciprocal
   */
  __QUALIFIER__ inline Vector3D rcp(void) const {
#ifdef __AVX__
    return Vector3D(_mm_div_pd(_mm_set1_pd(1.0), __vec), 1.0 / z);
#else
    return Vector3D(1.0 / x, 1.0 / y, 1.0 / z);
#endif
  }

  /**
   * Returns Euclidean length.
   */
  __QUALIFIER__ inline double norm( void ) const {
#ifdef __AVX__
    return sqrt(norm2());
#else
    return sqrt(x * x + y * y + z * z);
#endif
  }

  /**
   * Returns Euclidean length squared.
   */
  __QUALIFIER__ inline double norm2( void ) const {
#ifdef __AVX__
    return _mm_cvtsd_f64(_mm_dp_pd(__vec, __vec, 0b00110001)) + z * z;
#else
    return x * x + y * y + z * z;
#endif
  }

  /**
   * Returns unit vector.
   */
  __QUALIFIER__ inline Vector3D unit( void ) const {
    double rNorm = 1. / norm();
    return (*this) * rNorm;
  }

  /**
   * Divides by Euclidean length.
   */
  __QUALIFIER__ inline void normalize( void ) {
    (*this) /= norm();
  }

  __QUALIFIER__ inline Color toColor() const {
    return Color(r, g, b);
  }

  __QUALIFIER__ inline float illum() const {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
  }

  static Vector3D fromColor(const Color& c) {
    return Vector3D(c.r, c.g, c.b);
  }

}; // class Vector3D

// left scalar multiplication
__QUALIFIER__ inline Vector3D operator* ( const double& c, const Vector3D& v ) {
  return Vector3D( c * v.x, c * v.y, c * v.z );
}

// left scalar divide
__QUALIFIER__ inline Vector3D operator/(const double &c, const Vector3D &v) {
#ifdef __AVX__
  return Vector3D(_mm_div_pd(_mm_set1_pd(c), v.__vec), c / v._z);
#else
  return Vector3D(c / v.x, c / v.y, c / v.z);
#endif
}

// dot product (a.k.a. inner or scalar product)
__QUALIFIER__ inline double dot( const Vector3D& u, const Vector3D& v ) {
#ifdef __AVX__
  return _mm_cvtsd_f64(_mm_dp_pd(u.__vec, v.__vec, 0b00110001)) + u._z * v._z;
#else
  return u.x * v.x + u.y * v.y + u.z * v.z;
#endif
}

// cross product
__QUALIFIER__ inline Vector3D cross( const Vector3D& u, const Vector3D& v ) {
  return Vector3D( u.y*v.z - u.z*v.y,
                   u.z*v.x - u.x*v.z,
                   u.x*v.y - u.y*v.x );
}

// prints components
std::ostream& operator<<( std::ostream& os, const Vector3D& v );

} // namespace CGL

#endif // CGL_VECTOR3D_H
