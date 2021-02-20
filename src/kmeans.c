#include <stdio.h>
#include <stdlib.h>
#include <float.h> // FLT_MAX
#include <assert.h> // @TODO: implement our own assert

// #include "kmeans.h"
#include "types.h"
#include "logger.h"


void print_sample(float *sample, uint32 num_features) {
  printf("(");
  for (uint32 i = 0; i < num_features - 1; i++) {
    printf("%f, ", *(sample + i));
  }
  printf("%f", *(sample + num_features - 1));
  printf(")\n");
}

void print_samples(float *samples, uint32 num_samples, uint32 num_features) {
  for (uint32 i = 0; i < num_samples; i++) {
    float *sample_p = samples + num_features * i;
    print_sample(sample_p, num_features);
  }
}

float l2_norm(float *v1, float *v2, uint32 num_elements) {
  float dst = 0;
  for (int i = 0; i < num_elements; i++) {
    float dt = v1[i] - v2[i];
    dst += dt * dt;
  }
  return dst;
}

// @TODO: replace malloc with memory arena!

void k_means(float *dataset, uint32 num_samples, uint32 num_features, uint32 num_clusters, float tolerance, uint32 max_iterations, float *centroids_result, uint32 *groups_result) {
  
  //printf("dataset base pointer: %p\n", dataset);
  //printf("clusters: %u\n", num_clusters);
  //printf("samples: %u\n", num_samples);
  //printf("features: %u\n", num_features);
  //printf("tolerance: %f\n", tolerance);

  //printf("=== dataset ===\n");
  //print_samples(dataset, num_samples, num_features);
  //printf("\n");
  

  // initialize centroids to the values of the first num_clusters samples.
  size_t centroids_size = num_clusters * num_features * sizeof(float);
  float *centroids = (float *)malloc(centroids_size);
  
  for (uint32 r = 0; r < num_clusters; r++) {
    for (uint32 c = 0; c < num_features; c++) {
      uint32 index = r * num_features + c;
      centroids[index] = dataset[index]; 
    }
  } 

  uint32 *cluster_group_count = (uint32 *)malloc(num_clusters * sizeof(uint32));
  uint32 *cluster_groups = (uint32 *)malloc(num_samples * sizeof(uint32));
  float *new_centroids = (float *)malloc(num_clusters * num_features * sizeof(float)); 

  uint32 iteration = 1;
  while(1) {
    // printf("=== iteration %u ===\n", iteration);

    // reset new centroids + cluster group count
    for (uint32 c = 0; c < num_clusters; c++) {
      cluster_group_count[c] = 0;

      for (uint32 f = 0; f < num_features; f++) {
        uint32 index = c * num_features + f;
        new_centroids[index] = 0;
      }
    }

    for (uint32 s = 0; s < num_samples; s++) {
      float *sample = &dataset[s * num_features];

      float min_distance_metric = FLT_MAX;
      uint32 closest_centroid_id = -1;

      for (uint32 c = 0; c < num_clusters; c++) {
        float *centroid = &centroids[c * num_features]; 

        float distance_metric = l2_norm(sample, centroid, num_features); 

        if (distance_metric < min_distance_metric) {
          min_distance_metric = distance_metric;
          closest_centroid_id = c;
        }
      }

      assert(closest_centroid_id != -1);

      cluster_groups[s] = closest_centroid_id;
      cluster_group_count[closest_centroid_id] += 1;

      for (uint32 f = 0; f < num_features; f++) {
        float *new_centroid = &new_centroids[closest_centroid_id * num_features];
        new_centroid[f] += sample[f];
      }

      // printf("closest centroid: %d\n", closest_centroid_id);

    }

    float centroid_moved_by = 0;

    for (uint32 c = 0; c < num_clusters; c++) {
      float *new_centroid = &new_centroids[c * num_features];
      int group_count = cluster_group_count[c];

      for (uint32 f = 0; f < num_features; f++) {
        new_centroid[f] /= group_count;
      }

      float *centroid = &centroids[c * num_features];

      // calculate distance between new and old centroid
      float distance_metric = l2_norm(new_centroid, centroid, num_features);

      for (uint32 f = 0; f < num_features; f++) {
        centroid[f] = new_centroid[f];
      }

      if (centroid_moved_by < distance_metric) {
        centroid_moved_by = distance_metric;
      }
    }

    // printf("centroid moved by: %f\n", centroid_moved_by);
    if (centroid_moved_by < tolerance) break;
    if (max_iterations != 0 && iteration >= max_iterations) break;
    iteration += 1;
  }

  
  //printf("=== result ===\n");
  //print_samples(centroids, num_clusters, num_features);
  
  // @TODO: create a struct for the result.
  for (uint32 i = 0; i < num_samples; i++) {
    groups_result[i] = cluster_groups[i];
  } 

  for (uint32 c = 0; c < num_clusters; c++) {
    float *centroid = &centroids[c * num_features];
    float *centroid_result = &centroids_result[c * num_features];

    for (uint32 f = 0; f < num_features; f++) {
      centroid_result[f] = centroid[f];
    }
  }
  

  free(centroids);
  free(new_centroids);
  free(cluster_group_count);
  free(cluster_groups);
}

