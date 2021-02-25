#include <stdio.h>
#include <stdlib.h>
#include <float.h> // FLT_MAX
#include <assert.h> // @TODO: implement our own assert

// #include "kmeans.h"
#include "types.h"
#include "logger.h"


void print_sample(float *sample, uint32 n_features) {
  printf("(");
  for (uint32 i = 0; i < n_features - 1; i++) {
    printf("%f, ", *(sample + i));
  }
  printf("%f", *(sample + n_features - 1));
  printf(")\n");
}

void print_samples(float *samples, uint32 n_samples, uint32 n_features) {
  for (uint32 i = 0; i < n_samples; i++) {
    float *sample_p = samples + n_features * i;
    print_sample(sample_p, n_features);
  }
}

float l2_norm(float *v1, float *v2, uint32 n_elements) {
  float dst = 0;
  for (int i = 0; i < n_elements; i++) {
    float dt = v1[i] - v2[i];
    dst += dt * dt;
  }
  return dst;
}

// @TODO: replace malloc with memory arena!

void k_means(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float tolerance, uint32 max_iterations, float *centroids_result, uint32 *groups_result) {

  size_t centroids_size = n_clusters * n_features * sizeof(float);
  float *centroids = (float *)malloc(centroids_size);
  
  // initialize centroids to the values of the first n_clusters samples.
  for (uint32 c = 0; c < n_clusters; c++) {
    for (uint32 f = 0; f < n_features; f++) {
      uint32 index = c * n_features + f;
      centroids[index] = dataset[index]; 
    }
  } 

  uint32 *cluster_group_count = (uint32 *)malloc(n_clusters * sizeof(uint32));
  uint32 *cluster_groups = (uint32 *)malloc(n_samples * sizeof(uint32));
  float *new_centroids = (float *)malloc(n_clusters * n_features * sizeof(float)); 

  uint32 iteration = 1;
  while(1) {
   
    // reset new centroids + cluster group count
    for (uint32 c = 0; c < n_clusters; c++) {
      cluster_group_count[c] = 0;

      for (uint32 f = 0; f < n_features; f++) {
        uint32 index = c * n_features + f;
        new_centroids[index] = 0;
      }
    }

    // compute distances and assign samples to clusters
    for (uint32 s = 0; s < n_samples; s++) {
      float *sample = &dataset[s * n_features];

      float min_distance_metric = FLT_MAX;
      uint32 closest_centroid_id = -1;

      for (uint32 c = 0; c < n_clusters; c++) {
        float *centroid = &centroids[c * n_features]; 

        float distance_metric = l2_norm(sample, centroid, n_features); 

        if (distance_metric < min_distance_metric) {
          min_distance_metric = distance_metric;
          closest_centroid_id = c;
        }
      }

      assert(closest_centroid_id != -1);

      cluster_groups[s] = closest_centroid_id;
      cluster_group_count[closest_centroid_id] += 1;

      for (uint32 f = 0; f < n_features; f++) {
        float *new_centroid = &new_centroids[closest_centroid_id * n_features];
        new_centroid[f] += sample[f];
      }
    }

    // compute new centroid features and check whether the centroids have converged
    float centroid_moved_by = 0;
    for (uint32 c = 0; c < n_clusters; c++) {
      float *new_centroid = &new_centroids[c * n_features];
      int group_count = cluster_group_count[c];

      for (uint32 f = 0; f < n_features; f++) {
        new_centroid[f] /= group_count;
      }

      float *centroid = &centroids[c * n_features];

      // calculate distance between new and old centroid
      float distance_metric = l2_norm(new_centroid, centroid, n_features);

      for (uint32 f = 0; f < n_features; f++) {
        centroid[f] = new_centroid[f];
      }

      if (centroid_moved_by < distance_metric) {
        centroid_moved_by = distance_metric;
      }
    }

    if (centroid_moved_by < tolerance) break;
    if (max_iterations != 0 && iteration >= max_iterations) break;
    iteration += 1;
  }
  
  // @Speed: we could change the result before.
  for (uint32 i = 0; i < n_samples; i++) {
    groups_result[i] = cluster_groups[i];
  } 

  for (uint32 c = 0; c < n_clusters; c++) {
    float *centroid = &centroids[c * n_features];
    float *centroid_result = &centroids_result[c * n_features];

    for (uint32 f = 0; f < n_features; f++) {
      centroid_result[f] = centroid[f];
    }
  }
  
  free(centroids);
  free(new_centroids);
  free(cluster_group_count);
  free(cluster_groups);
}

