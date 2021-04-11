#include <stdio.h>
#include <stdlib.h>
#include <float.h> // DBL_MAX
#include <string.h> // memcpy
#include <assert.h>
#include <queue>
#include <vector>

#include "types.h"
#include "logger.h"
#include "timer.h"


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

inline double squared_euclidian_distance(float *v1, float *v2, uint32 n_elements) {
  double dst = 0;
  for (uint32 i = 0; i < n_elements; i++) {
    float dt = v1[i] - v2[i];
    dst += dt * dt;
  }
  return dst;
}

float *init_centroids_to_first(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters) {
  size_t centroids_size = n_clusters * n_features * sizeof(float);
  float *centroids = (float *)malloc(centroids_size);
  
  // initialize centroids to the values of the first n_clusters samples.
  for (uint32 c = 0; c < n_clusters; c++) {
    for (uint32 f = 0; f < n_features; f++) {
      uint32 index = c * n_features + f;
      centroids[index] = dataset[index]; 
    }
  }

  return centroids;
  
}

// @TODO: replace malloc with memory arena!

void k_means(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float tolerance, uint32 max_iterations, float *centroid_init, float *centroids_result, uint32 *groups_result) {
  size_t centroids_size = n_clusters * n_features * sizeof(float);
  float *centroids = (float *)malloc(centroids_size);

  start();

  for (uint32 c = 0; c < n_clusters; c++) {
    for (uint32 f = 0; f < n_features; f++) {
      uint32 index = c * n_features + f;
      centroids[index] = centroid_init[index]; 
    }
  }

  stop();
  //printf("centroid copy took %f s\n", elapsed());


  //print_samples(centroids, n_clusters, n_features);

  start();

  uint32 *cluster_sizes = (uint32 *)malloc(n_clusters * sizeof(uint32));
  uint32 *cluster_groups = (uint32 *)malloc(n_samples * sizeof(uint32));
  
  float *new_centroids = (float *)malloc(centroids_size); 

  stop();
  //printf("malloc took %f s\n", elapsed());


  uint32 iteration = 1;
  while(1) {

    //start();
    // reset new centroids + cluster sizes
    for (uint32 c = 0; c < n_clusters; c++) {
      cluster_sizes[c] = 0;

      for (uint32 f = 0; f < n_features; f++) {
        uint32 index = c * n_features + f;
        new_centroids[index] = 0;
      }
    }

    //stop();
    //printf("%d reset new centroids %f s\n", iteration, elapsed());

    //start();
  
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

      cluster_groups[s] = closest_centroid_id;
      cluster_sizes[closest_centroid_id] += 1;

      for (uint32 f = 0; f < n_features; f++) {
        float *new_centroid = new_centroids + closest_centroid_id * n_features;
        new_centroid[f] += sample[f];
      }
    }
    
    //stop();
    //printf("%d assignment %f s\n", iteration, elapsed());

    //start();

    // compute new centroid features and check whether the centroids have converged
    double centroid_moved_by = 0;
    for (uint32 c = 0; c < n_clusters; c++) {
      float *centroid = centroids + c * n_features;
      float *new_centroid = new_centroids + c * n_features;
      uint32 group_count = cluster_sizes[c];

      for (uint32 f = 0; f < n_features; f++) {
        new_centroid[f] /= group_count;
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

    //stop();
    //printf("%d centroid update + done check %f s\n", iteration, elapsed());

    //print_samples(centroids, n_clusters, n_features);    
    //printf("\n");

    if (centroid_moved_by < tolerance) {
      //printf("centroids converged %f\n", centroid_moved_by);
      break;
    }
    
    if (max_iterations != 0 && iteration >= max_iterations) {
      printf("centroids did not converge!\n");
      break;
    }
    
    iteration += 1;
  }

  stop();
  printf("iterations took %f s\n", elapsed());
  printf("iterations count: %d\n", iteration);

  
  if (groups_result != NULL) {  
    for (uint32 i = 0; i < n_samples; i++) {
      groups_result[i] = cluster_groups[i];
    } 
  }

  if (centroids_result != NULL) {
    for (uint32 c = 0; c < n_clusters; c++) {
      float *centroid = &centroids[c * n_features];
      float *centroid_result = &centroids_result[c * n_features];

      for (uint32 f = 0; f < n_features; f++) {
        centroid_result[f] = centroid[f];
      }
    }
  }
  
  free(centroids);
  free(new_centroids);
  free(cluster_sizes);
  free(cluster_groups);
}



struct ClusterJob {
  float *samples;
  uint32 n_samples;
  
  uint32 layer;
  uint32 id;
  
  uint32 *mask_indices;
};

uint32 count(uint32 target, uint32 *arr, uint32 arr_len) {
  uint32 count = 0;
  
  for (uint32 i = 0; i < arr_len; i++) {
    if (arr[i] == target) count += 1;
  }
  
  return count;
}

void update_mask(uint32 *mask, uint32 *groups, uint32 id, uint32 *mask_indices, uint32 n_samples) {
  for (uint32 i = 0; i < n_samples; i++) {
    mask[mask_indices[i]] = id + groups[i];
  }
}

void update_mask2(uint32 *mask, uint32 id, uint32 *mask_indices, uint32 n_samples) {
  for (uint32 i = 0; i < n_samples; i++) {
    mask[mask_indices[i]] = id;
  }
}

ClusterJob create_child_clusterjob(uint32 group, uint32 *child_groups, ClusterJob *parent, uint32 n_features) {
  uint32 group_size = count(group, child_groups, parent->n_samples);
  
  ClusterJob child;
  child.n_samples = group_size;
  child.layer = parent->layer + 1;
  child.mask_indices = (uint32 *)malloc(group_size * sizeof(uint32));
  child.samples = (float *)malloc(group_size * n_features * sizeof(float));
  child.id = 2*parent->id + group;
  
  uint32 i = 0;
  for (uint32 j = 0; j < parent->n_samples; j++) {
    if (child_groups[j] == group) {
      child.mask_indices[i] = parent->mask_indices[j];
      
      for (uint32 k = 0; k < n_features; k++) {
        child.samples[i*n_features + k] = parent->samples[j*n_features + k];
      }
      
      i += 1;
    }
  }
  
  return child;
}

void fractal_k_means(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_layers, uint32 *result) {
  uint32 *mask = (uint32 *)malloc(n_samples * sizeof(uint32));
  
  for (uint32 i = 0; i < n_samples; i++) {
    mask[i] = 0;
  }
  
  ClusterJob root;
  root.samples = dataset;
  root.n_samples = n_samples;
  root.layer = 0;
  root.id = 0;
  root.mask_indices = (uint32 *)malloc(n_samples * sizeof(uint32));
  
  for (uint32 i = 0; i < n_samples; i++) {
    root.mask_indices[i] = i;
  }
  
  uint32 layer = 0;
  
  // @TODO: use a circular buffer instead. This is only possible if we state the number of layers or clusters as input.
  std::queue<ClusterJob> queue;
  
  queue.push(root);
  
  while (queue.size() > 0) {
    ClusterJob current = queue.front();
    queue.pop();
      
    if (current.layer > layer) {
      layer += 1;

      for (uint32 i = 0; i < n_samples; i++) {
        result[n_samples * (current.layer-1) + i] = mask[i];
      }
    }
    
    if (current.layer < n_layers) {
      if (current.n_samples >= 2) {
        uint32 *groups = (uint32 *)malloc(current.n_samples*sizeof(uint32));
        k_means(current.samples, current.n_samples, n_features, 2, 0.001f, 100, NULL, groups);
      
        update_mask(mask, groups, 2*current.id, current.mask_indices, current.n_samples);
        
        ClusterJob child1 = create_child_clusterjob(0, groups, &current, n_features);
        queue.push(child1);
        
        ClusterJob child2 = create_child_clusterjob(1, groups, &current, n_features);
        queue.push(child2);
        
        free(groups);
        
      } else {
        update_mask2(mask, 2*current.id, current.mask_indices, current.n_samples);
      }
    }
    
    free(current.mask_indices);
    if (current.layer != 0) free(current.samples);
  }
  
  for (uint32 i = 0; i < n_samples; i++) {
    result[n_samples * (n_layers-1) + i] = mask[i];
  }
  
  free(mask);
  
}

