#include <stdlib.h>
#include <assert.h>

#include "initialization_procedures.h"

#define RANDOM(min, max) (((max) - (min)) * (rand() / (double) RAND_MAX) + (min))

inline double squared_euclidian_distance(float *v1, float *v2, uint32 n_elements) {
  double dst = 0;
  for (uint32 i = 0; i < n_elements; i++) {
    float dt = v1[i] - v2[i];
    dst += dt * dt;
  }
  return dst;
}

void init_centroids_randomly(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids) {
  uint32 range = n_samples / n_clusters;

  for (uint32 c = 0; c < n_clusters; c++) {
    uint32 sample = (uint32) RANDOM(c * range, (c + 1) * range - 1);
    
    for (uint32 f = 0; f < n_features; f++) {
      uint32 i = c * n_features + f;
      uint32 j = sample * n_features + f;
      
      centroids[i] = dataset[j]; 
    }
  }
}

void init_centroids_using_kmeansplusplus(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids) {
  // centroid 1
  uint32 sample = (uint32) RANDOM(0, n_samples - 1);
  for (uint32 f = 0; f < n_features; f++) {
      uint32 i = f;
      uint32 j = sample * n_features + f;
      centroids[i] = dataset[j];
  }

  float *distances = (float *)malloc(n_samples * sizeof(float));
  for (uint32 i = 0; i < n_samples; i++) {
    distances[i] = FLT_MAX;
  }

  // remaining centroids
  for (uint32 c = 0; c < n_clusters - 1; c++) {
    float *centroid = centroids + c * n_features;

    // compute the distance to the closest centroid for each point.
    for (uint32 s = 0; s < n_samples; s++) {
      float *sample = dataset + s * n_features;

      float distance = (float) squared_euclidian_distance(centroid, sample, n_features);
      if (distance < distances[s]) distances[s] = distance;
    }

    uint32 best = -1;
    uint32 max_distance = 0;
    for (uint32 s = 0; s < n_samples; s++) {
      if (distances[s] > max_distance) {
        max_distance = distances[s];
        best = s;
      }
    }

    assert(best != -1);

    // set centroid C to the sample furthest away from all currently selected centroids.
    for (uint32 f = 0; f < n_features; f++) {
        uint32 i = (c + 1) * n_features + f;
        uint32 j = best * n_features + f;
        centroids[i] = dataset[j];
    }
  }

  free(distances);

}

void init_centroids_to_first_samples(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids) {
  // initialize centroids to the values of the first n_clusters samples.
  for (uint32 c = 0; c < n_clusters; c++) {
    for (uint32 f = 0; f < n_features; f++) {
      uint32 index = c * n_features + f;
      centroids[index] = dataset[index]; 
    }
  }  
}
