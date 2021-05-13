#include <stdio.h>
#include <stdlib.h>
#include <float.h> // DBL_MAX
#include <string.h> // memcpy, memset
#include <assert.h>
#include <omp.h>
#include <math.h>

#include "arena.h"
#include "kmeans_wcs.h"
#include "distances.h"

#define MIN_DIMENSION_SIZE_PER_THREAD 1000

// If EPSILON becomes too small, then the centroid will not move, because the initial position of the centroid is on top of an observation. 
// Thus, this observation will have "too much" weight when updating the centroid position and it will pull the centroid to stay where it is. 
// So the centroids will just stay at there initial position.
#define EPSILON 0.01 // this value seems to give enough mobility.

static
void assign_samples_to_clusters(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids, uint32 *clusters, uint32 *cluster_sizes, float *distances) {
  int dimension = (n_samples * n_features * n_clusters) / MIN_DIMENSION_SIZE_PER_THREAD;
  int s;
 
  #pragma omp parallel for if (dimension > MIN_DIMENSION_SIZE_PER_THREAD)
  for (s = 0; s < n_samples; s++) {
    float *sample = dataset + s * n_features;

    float min_distance = FLT_MAX;
    uint32 closest_centroid_id = -1;

    for (uint32 c = 0; c < n_clusters; c++) {
      float *centroid = centroids + c * n_features; 

      float distance = squared_euclidian_distance(sample, centroid, n_features); 

      if (distance < min_distance) {
        min_distance = distance;
        closest_centroid_id = c;
      }
    }

    assert(closest_centroid_id != -1);

    clusters[s] = closest_centroid_id;
    distances[s] = (float) sqrt(min_distance) + EPSILON;
  }

}

static 
float update_centroids(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, uint32 *clusters, float *centroids, float *new_centroids, float *lambda_invs, float *distances, uint32 *cluster_sizes) {
  float change = 0.0;

  for (uint32 s = 0; s < n_samples; s++) {
    uint32 closest_centroid_id = clusters[s];
    cluster_sizes[closest_centroid_id] += 1;

    float distance = distances[s];
    assert(distance != 0);

    lambda_invs[closest_centroid_id] += 1. / distance;

    float *sample = dataset + s * n_features;

    for (uint32 f = 0; f < n_features; f++) {
      float *new_centroid = new_centroids + closest_centroid_id * n_features;
      new_centroid[f] += sample[f] / distance;
    }
  }

  // compute new centroid features and check whether the centroids have converged
  for (uint32 c = 0; c < n_clusters; c++) {
    float *centroid = centroids + c * n_features;
    float *new_centroid = new_centroids + c * n_features;

    float lambda_inv = lambda_invs[c];
    assert(lambda_inv != 0);

    for (uint32 f = 0; f < n_features; f++) {
      new_centroid[f] /= lambda_inv;
    }
    
    // calculate distance between new and old centroid
    float distance_metric = squared_euclidian_distance(new_centroid, centroid, n_features);

    for (uint32 f = 0; f < n_features; f++) {
      centroid[f] = new_centroid[f];
    }

    if (distance_metric > change) {
      change = distance_metric;
    }
  }

  return change;

}

Buffer kmeans_wcs_allocate_buffer(uint32 n_samples, uint32 n_features, uint32 n_clusters) {
  size_t centroids_size = n_clusters * n_features * sizeof(float);
  size_t cluster_sizes_size = n_clusters * sizeof(uint32);
  size_t clusters_size = n_samples * sizeof(uint32);
  size_t new_centroids_size = centroids_size;
  size_t lambda_invs_size = n_clusters * sizeof(float);
  size_t distances_size = n_samples * sizeof(float);

  size_t buffer_size = centroids_size + cluster_sizes_size + clusters_size + new_centroids_size + lambda_invs_size + distances_size + DEFAULT_ALIGNMENT * 6;
  
  void *memory = malloc(buffer_size);

  Buffer buffer;
  buffer.size = buffer_size;
  buffer.memory = memory;

  return buffer;
}

void kmeans_wcs_algorithm(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, 
                          float tolerance, uint32 max_iterations, float *centroid_init, float *centroids_result, uint32 *groups_result, 
                          uint32 *converged_result, Buffer *buffer) {

  size_t centroids_size = n_clusters * n_features * sizeof(float);
  size_t cluster_sizes_size = n_clusters * sizeof(uint32);
  size_t clusters_size = n_samples * sizeof(uint32);
  size_t new_centroids_size = centroids_size;
  size_t lambda_invs_size = n_clusters * sizeof(float);
  size_t distances_size = n_samples * sizeof(float);

  size_t buffer_size = centroids_size + cluster_sizes_size + clusters_size + new_centroids_size + lambda_invs_size + distances_size + DEFAULT_ALIGNMENT * 6;
  assert(buffer->size >= buffer_size);

  Arena arena = {0};
  arena_init(&arena, buffer->memory, buffer_size);

  float *centroids = (float *)arena_alloc(&arena, centroids_size);
  uint32 *cluster_sizes = (uint32 *)arena_alloc(&arena, cluster_sizes_size);
  uint32 *clusters = (uint32 *)arena_alloc(&arena, clusters_size);
  float *new_centroids = (float *)arena_alloc(&arena, new_centroids_size);
  float *lambda_invs = (float *)arena_alloc(&arena, lambda_invs_size);
  float *distances = (float *)arena_alloc(&arena, distances_size);

  memcpy(centroids, centroid_init, centroids_size);

  uint32 converged = 1;

  uint32 iteration = 1;
  while(1) {
    // reset new centroids + cluster sizes
    memset(cluster_sizes, 0, cluster_sizes_size);
    memset(new_centroids, 0, new_centroids_size);
    memset(lambda_invs, 0, lambda_invs_size);

    assign_samples_to_clusters(dataset, n_samples, n_features, n_clusters, centroids, clusters, cluster_sizes, distances);
    
    float change = update_centroids(dataset, n_samples, n_features, n_clusters, clusters, centroids, new_centroids, lambda_invs, distances, cluster_sizes);

    if (change < tolerance) break;
    
    if (max_iterations != 0 && iteration >= max_iterations) {
      converged = 0;
      break;
    }
    
    iteration += 1;
  }

  if (groups_result != NULL) 
    memcpy(groups_result, clusters, clusters_size);

  if (centroids_result != NULL) 
    memcpy(centroids_result, centroids, centroids_size);

  if (converged_result != NULL)
    *converged_result = converged;

  arena_free_all(&arena);

}
