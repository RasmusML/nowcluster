#include <stdio.h>
#include <stdlib.h>
#include <float.h> // DBL_MAX
#include <string.h> // memcpy, memset
#include <assert.h>
#include <omp.h>

#include "arena.h"
#include "kmeans.h"
#include "initialization_procedures.h"

// https://docs.microsoft.com/en-us/cpp/parallel/openmp/reference/openmp-directives?view=msvc-160#for-openmp

inline double squared_euclidian_distance(float *v1, float *v2, uint32 n_elements) {
  double dst = 0;
  for (uint32 i = 0; i < n_elements; i++) {
    float dt = v1[i] - v2[i];
    dst += dt * dt;
  }
  return dst;
}

#define MIN_DIMENSION_SIZE_PER_THREAD 1000

static
void assign_samples_to_clusters_single_threaded(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids, uint32 *clusters, uint32 *cluster_sizes) {
  for (int s = 0; s < n_samples; s++) {
    float *sample = dataset + s * n_features;

    double min_distance = DBL_MAX;
    uint32 closest_centroid_id = -1;

    for (uint32 c = 0; c < n_clusters; c++) {
      float *centroid = centroids + c * n_features; 

      double distance = squared_euclidian_distance(sample, centroid, n_features); 

      if (distance < min_distance) {
        min_distance = distance;
        closest_centroid_id = c;
      }
    }

    assert(closest_centroid_id != -1);

    clusters[s] = closest_centroid_id;
    cluster_sizes[closest_centroid_id] += 1;
 
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

      double min_distance = DBL_MAX;
      uint32 closest_centroid_id = -1;

      for (uint32 c = 0; c < n_clusters; c++) {
        float *centroid = centroids + c * n_features; 

        double distance = squared_euclidian_distance(sample, centroid, n_features); 

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
  for (int32 s = 0; s < n_samples; s++) {
    uint32 closest_centroid_id = clusters[s];
    cluster_sizes[closest_centroid_id] += 1;
  }

}

int NUM_THREADS;

static
void assign_samples_to_clusters(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids, uint32 *clusters, uint32 *cluster_sizes) {

  int num_threads = (n_samples * n_features) / MIN_DIMENSION_SIZE_PER_THREAD;

  if (num_threads >= 4) {
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
    float *sample = dataset + s * n_features;

    uint32 closest_centroid_id = clusters[s];

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
      change = (float) distance_metric;
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
                      float tolerance, uint32 max_iterations, float *centroid_init, uint8 use_wcss,
                      float *centroids_result, uint32 *groups_result, uint32 *converged_result, Buffer *buffer) {


  int NUM_THREADS = omp_get_num_threads();

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

/*
void kmeans_algorithm_old(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, 
                      float tolerance, uint32 max_iterations, float *centroid_init, const bool use_wcss,
                      float *centroids_result, uint32 *groups_result, uint32 *converged_result) {

  size_t centroids_size = n_clusters * n_features * sizeof(float);
  size_t cluster_sizes_size = n_clusters * sizeof(uint32);
  size_t clusters_size = n_samples * sizeof(uint32);
  size_t new_centroids_size = centroids_size;

  size_t lambda_inv_size = n_clusters * sizeof(float);

  float *centroids;
  uint32 *cluster_sizes;
  uint32 *clusters;
  float *new_centroids;

  centroids = (float *)malloc(centroids_size);
  cluster_sizes = (uint32 *)malloc(cluster_sizes_size);
  clusters = (uint32 *)malloc(clusters_size);
  new_centroids = (float *)malloc(new_centroids_size); 

  memcpy(centroids, centroid_init, centroids_size);

  float *lambda_inv;
  if (!use_wcss) lambda_inv = (float *)malloc(lambda_inv_size);

  uint32 converged = 1;
  
  uint32 iteration = 1;
  while(1) {
    // reset new centroids + cluster sizes
    memset(cluster_sizes, 0, cluster_sizes_size);
    memset(new_centroids, 0, new_centroids_size);

    if (!use_wcss) {
      memset(lambda_inv, 0, lambda_inv_size);
    }

    // compute distances and assign samples to clusters
    for (uint32 s = 0; s < n_samples; s++) {
      float *sample = dataset + s * n_features;

      double min_distance = DBL_MAX;
      uint32 closest_centroid_id = -1;

      for (uint32 c = 0; c < n_clusters; c++) {
        float *centroid = centroids + c * n_features; 

        double distance = squared_euclidian_distance(sample, centroid, n_features); 

        if (distance < min_distance) {
          min_distance = distance;
          closest_centroid_id = c;
        }
      }

      assert(closest_centroid_id != -1);

      clusters[s] = closest_centroid_id;
      cluster_sizes[closest_centroid_id] += 1;

      for (uint32 f = 0; f < n_features; f++) {
        float *new_centroid = new_centroids + closest_centroid_id * n_features;

        if (use_wcss) {
          new_centroid[f] += sample[f];
        } else {
          if (min_distance == 0) {
            //new_centroid[f] += sample[f];
          } else {
            new_centroid[f] += sample[f] / (float) min_distance;
          }
        }
      }

      // @TODO: figure out what do if min_distance is 0 or very close to 0.
      if (!use_wcss) {
        if (min_distance == 0) {  
          //lambda_inv[closest_centroid_id] += 1;
        } else {
          lambda_inv[closest_centroid_id] += 1 / (float) min_distance;
        }
      }
    }

    // compute new centroid features and check whether the centroids have converged
    double centroid_moved_by = 0; 
    for (uint32 c = 0; c < n_clusters; c++) {
      float *centroid = centroids + c * n_features;
      float *new_centroid = new_centroids + c * n_features;
      uint32 group_count = cluster_sizes[c];

      if (group_count == 0) continue;

      for (uint32 f = 0; f < n_features; f++) {
        if (use_wcss) {
          new_centroid[f] /= group_count;
        } else {
          new_centroid[f] /= lambda_inv[c];
        }
      }

      // calculate distance between new and old centroid
      double distance_metric = squared_euclidian_distance(new_centroid, centroid, n_features);

      for (uint32 f = 0; f < n_features; f++) {
        centroid[f] = new_centroid[f];
      }

      if (centroid_moved_by < distance_metric) {
        centroid_moved_by = distance_metric;
      }
    }

    printf("moved by %f\n", centroid_moved_by);
    if (centroid_moved_by < tolerance) {
      break;
    }
    
    if (max_iterations != 0 && iteration >= max_iterations) {
      converged = 0;
      break;
    }
    
    iteration += 1;
  }

  if (groups_result != NULL) {  
    memcpy(groups_result, clusters, clusters_size);
  }

  if (centroids_result != NULL) {
    memcpy(centroids_result, centroids, centroids_size);
  }

  if (converged_result != NULL) {
    *converged_result = converged;
  }

  if (!use_wcss) free(lambda_inv); 

  free(centroids);
  free(cluster_sizes);
  free(clusters);
  free(new_centroids);

}

*/

/*


    
    std::queue<std::thread> queue;
    for (uint32 i = 0; i < processor_count - 1; i++) {
      memset(jobs[i], 0, cluster_sizes_size);

      uint32 offset = i * samples_per_thread;
      
      //printf("offset: %u, samples_per_thread: %u\n", offset, samples_per_thread);
      //assign_samples_to_clusters(dataset + i * samples_per_thread, samples_per_thread, n_features, n_clusters, centroids, clusters, cluster_sizes);
      std::thread thread(assign_samples_to_clusters, &dataset[offset], 
                         samples_per_thread, n_features, n_clusters, centroids, &clusters[offset], jobs[i]);

      queue.push(std::move(thread));
    }

    //assign_samples_to_clusters(dataset + (processor_count - 1) * samples_per_thread, samples_per_thread + samples_rest, n_features, n_clusters, centroids, clusters, cluster_sizes);

    memset(jobs[processor_count - 1], 0, cluster_sizes_size);
    
    uint32 offset = (processor_count - 1) * samples_per_thread;
    printf("offset: %u, samples_per_thread: %u\n", offset, samples_per_thread);
    std::thread thread(assign_samples_to_clusters, dataset + offset, 
                       samples_per_thread + samples_rest, n_features, n_clusters, centroids, clusters + offset, jobs[processor_count - 1]);
    queue.push(std::move(thread));
    
    for (uint32 i = 0; i < processor_count; i++) {
      std::thread& thread = queue.front();
      thread.join();
      queue.pop();

      uint32 *job = jobs[i];
      for (uint32 c = 0; c < n_clusters; c++) {
        cluster_sizes[c] += job[c];
      }
    }

    printf("\nall thread joined\n");

    for (uint32 c = 0; c < n_clusters; c++) {
        printf("%u:%u ", c, cluster_sizes[c]);
    }
    printf("\n");
    
    memset(cluster_sizes, 0, cluster_sizes_size);
    memset(new_centroids, 0, new_centroids_size);
    
    */