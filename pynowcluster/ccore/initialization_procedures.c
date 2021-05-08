#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <float.h> // DBL_MAX

#include "initialization_procedures.h"

const uint32 INIT_PROVIDED = 0;
const uint32 INIT_RANDOMLY = 1;
const uint32 INIT_KMEANS_PLUS_PLUS = 2;
const uint32 INIT_TO_FIRST_SAMPLES = 3;

void init_centroids(uint32 init_method, float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroid_init, float *custom_centroid_init) {
  if (init_method == INIT_PROVIDED) memcpy(centroid_init, custom_centroid_init, n_clusters * n_features * sizeof(float));
  else if (init_method == INIT_KMEANS_PLUS_PLUS) init_centroids_using_kmeansplusplus(dataset, n_samples, n_features, n_clusters, centroid_init);
  else if (init_method == INIT_RANDOMLY) init_centroids_randomly(dataset, n_samples, n_features, n_clusters, centroid_init);     
  else if (init_method == INIT_TO_FIRST_SAMPLES) init_centroids_to_first_samples(dataset, n_samples, n_features, n_clusters, centroid_init);     
  else assert(0);
}

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
