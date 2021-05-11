#include "distances.h"

float squared_euclidian_distance(float *v1, float *v2, uint32 n_elements) {
  float dst = 0;
  for (uint32 i = 0; i < n_elements; i++) {
    float dt = v1[i] - v2[i];
    dst += dt * dt;
  }
  return dst;
}