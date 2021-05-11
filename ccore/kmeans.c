#include <stdio.h>
#include <stdlib.h>
#include <float.h> // DBL_MAX
#include <string.h> // memcpy, memset
#include <assert.h>
#include <omp.h>

#include "arena.h"
#include "kmeans.h"
#include "distances.h"

// https://docs.microsoft.com/en-us/cpp/parallel/openmp/reference/openmp-directives?view=msvc-160#for-openmp


#define MIN_DIMENSION_SIZE_PER_THREAD 1000

static
void assign_samples_to_clusters_single_threaded(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids, uint32 *clusters, uint32 *cluster_sizes) {
  for (int s = 0; s < n_samples; s++) {
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
  }
}

static
void assign_samples_to_clusters_multi_threaded(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids, uint32 *clusters, uint32 *cluster_sizes, int num_threads) {
  omp_set_num_threads(num_threads);

  int s;

  #pragma omp parallel
  { 
    #pragma omp for
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
    }
  }
  
  //const int ithread = omp_get_thread_num();  
}

static int NUM_THREADS;

static
void assign_samples_to_clusters(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids, uint32 *clusters, uint32 *cluster_sizes) {

  int num_threads = (n_samples * n_features) / MIN_DIMENSION_SIZE_PER_THREAD;

  if (num_threads >= 2) {
    if (num_threads > NUM_THREADS) num_threads = NUM_THREADS;
    assign_samples_to_clusters_multi_threaded(dataset, n_samples, n_features, n_clusters, centroids, clusters, cluster_sizes, num_threads);
        
  } else {
    assign_samples_to_clusters_single_threaded(dataset, n_samples, n_features, n_clusters, centroids, clusters, cluster_sizes);
  } 
}

static 
float update_centroids(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, uint32 *clusters, float *centroids, float *new_centroids, uint32 *cluster_sizes) {
  float change = 0.0f;

  for (uint32 s = 0; s < n_samples; s++) {
    uint32 closest_centroid_id = clusters[s];
    cluster_sizes[closest_centroid_id] += 1;

    float *sample = dataset + s * n_features;

    for (uint32 f = 0; f < n_features; f++) {
      float *new_centroid = new_centroids + closest_centroid_id * n_features;
      new_centroid[f] += sample[f];
    }
  }

  // compute new centroid features and check whether the centroids have converged
  for (uint32 c = 0; c < n_clusters; c++) {
    float *centroid = centroids + c * n_features;
    float *new_centroid = new_centroids + c * n_features;
    uint32 group_count = cluster_sizes[c];

    if (group_count == 0) continue;

    for (uint32 f = 0; f < n_features; f++) {
      new_centroid[f] /= group_count;
    }
    
    // calculate distance between new and old centroid
    double distance_metric = squared_euclidian_distance(new_centroid, centroid, n_features);

    for (uint32 f = 0; f < n_features; f++) {
      centroid[f] = new_centroid[f];
    }

    if (distance_metric > change) {
      change = distance_metric;
    }
  }

  return change;

}

Buffer kmeans_allocate_buffer(uint32 n_samples, uint32 n_features, uint32 n_clusters) {
  size_t centroids_size = n_clusters * n_features * sizeof(float);
  size_t cluster_sizes_size = n_clusters * sizeof(uint32);
  size_t clusters_size = n_samples * sizeof(uint32);
  size_t new_centroids_size = centroids_size;

  size_t buffer_size = centroids_size + cluster_sizes_size + clusters_size + new_centroids_size + DEFAULT_ALIGNMENT * 4;
  
  void *memory = malloc(buffer_size);

  Buffer buffer;
  buffer.size = buffer_size;
  buffer.memory = memory;

  return buffer;
}

void kmeans_algorithm(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, 
                      float tolerance, uint32 max_iterations, float *centroid_init, float *centroids_result, 
                      uint32 *groups_result, uint32 *converged_result, Buffer *buffer) {


  NUM_THREADS = omp_get_num_threads();

  size_t centroids_size = n_clusters * n_features * sizeof(float);
  size_t cluster_sizes_size = n_clusters * sizeof(uint32);
  size_t clusters_size = n_samples * sizeof(uint32);
  size_t new_centroids_size = centroids_size;

  size_t buffer_size = centroids_size + cluster_sizes_size + clusters_size + new_centroids_size + DEFAULT_ALIGNMENT * 4;
  assert(buffer->size >= buffer_size);

  Arena arena = {0};
  arena_init(&arena, buffer->memory, buffer_size);

  float *centroids = (float *)arena_alloc(&arena, centroids_size);
  uint32 *cluster_sizes = (uint32 *)arena_alloc(&arena, cluster_sizes_size);
  uint32 *clusters = (uint32 *)arena_alloc(&arena, clusters_size);
  float *new_centroids = (float *)arena_alloc(&arena, new_centroids_size);

  memcpy(centroids, centroid_init, centroids_size);

  uint32 converged = 1;
  
  uint32 iteration = 1;
  while(1) {
    // reset new centroids + cluster sizes
    memset(cluster_sizes, 0, cluster_sizes_size);
    memset(new_centroids, 0, new_centroids_size);

    assign_samples_to_clusters(dataset, n_samples, n_features, n_clusters, centroids, clusters, cluster_sizes);
    
    float change = update_centroids(dataset, n_samples, n_features, n_clusters, clusters, centroids, new_centroids, cluster_sizes);

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
