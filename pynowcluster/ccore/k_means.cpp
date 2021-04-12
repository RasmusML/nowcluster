#include <stdio.h>
#include <stdlib.h>
#include <float.h> // DBL_MAX
#include <string.h> // memcpy
#include <assert.h>
#include <queue>
#include <vector>
#include <list>

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

void init_centroids_to_first(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids) {
  // initialize centroids to the values of the first n_clusters samples.
  for (uint32 c = 0; c < n_clusters; c++) {
    for (uint32 f = 0; f < n_features; f++) {
      uint32 index = c * n_features + f;
      centroids[index] = dataset[index]; 
    }
  }  
}

// @TODO: free cluster_init, if we do the malloc!.

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
  //printf("iterations took %f s\n", elapsed());
  //printf("iterations count: %d\n", iteration);

  
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
  uint32 layer;
  uint32 n_samples;
  uint32 mask_indices_start;
};

void update_mask(uint32 id, uint32 *mask, uint32 *mask_indices, uint32 mask_indices_start, uint32 n_samples) {
  for (uint32 i = 0; i < n_samples; i++) {
    uint32 mask_index = mask_indices_start + i;
    mask[mask_indices[mask_index]] = id;
  }
}

void update_mask_by_offset(uint32 *offsets, uint32 id, uint32 *mask, uint32 *mask_indices, uint32 mask_indices_start, uint32 n_samples) {
  for (uint32 i = 0; i < n_samples; i++) {
    uint32 mask_index = mask_indices_start + i;
    mask[mask_indices[mask_index]] = id + offsets[i];
  }
}

#define SWAP(x, y, T) do { T _SWAP = x; x = y; y = _SWAP; } while (0)

ClusterJob create_child_clusterjob(uint32 cluster, uint32 *clusters, uint32 mask_indices_at, uint32* mask_indices, ClusterJob *parent, uint32 n_features) {
  uint32 offset = mask_indices_at - parent->mask_indices_start;
  uint32 i = offset;

  for (uint32 j = offset; j < parent->n_samples; j++) {

    if (clusters[j] == cluster) {
      SWAP(clusters[i], clusters[j], uint32);
      SWAP(mask_indices[parent->mask_indices_start + i], mask_indices[parent->mask_indices_start + j], uint32);
      
      i += 1;
    }
  }

  ClusterJob child;
  child.n_samples = i - offset;
  child.layer = parent->layer + 1;
  child.mask_indices_start = mask_indices_at;

  return child;
}

std::list<uint32 *> fractal_result;

void fractal_k_means(float *dataset, uint32 n_samples, uint32 n_features, float tolerance, uint32 max_iterations, uint32 *layers_result) {  
  uint32 *mask = (uint32 *) malloc(n_samples * sizeof(uint32));
  uint32 *mask_indices = (uint32 *) malloc(n_samples * sizeof(uint32));

  for (uint32 i = 0; i < n_samples; i++) {
    mask_indices[i] = i;
  }

  uint32 *clusters = (uint32 *) malloc(n_samples * sizeof(uint32));
  float *samples = (float *) malloc(n_samples * n_features * sizeof(float));

  const uint32 max_centroids = 2;
  float *centroid_inits = (float *) malloc(max_centroids * n_features * sizeof(float));
  
  ClusterJob root;
  root.n_samples = n_samples;
  root.layer = 0;
  root.mask_indices_start = 0;
  
  uint32 layer = 0;
  uint32 cluster_id = 0;

  uint8 splitting = 0;
  
  std::queue<ClusterJob> queue;
  
  queue.push(root);
  
  while (queue.size() > 0) {
    ClusterJob current = queue.front();
    queue.pop();
      
    if (current.layer > layer) {
      if (splitting == 0) break;
      splitting = 0;

      cluster_id = 0;
      layer += 1;

      uint32 *mask_result = (uint32 *) malloc(n_samples * sizeof(uint32));
      for (uint32 i = 0; i < n_samples; i++) {
        mask_result[i] = mask[i];
      }

      fractal_result.push_back(mask_result);
    }
    
    if (current.n_samples >= 2) {
      splitting = 1;

      for (uint32 i = 0; i < current.n_samples; i++) {
        uint32 index = mask_indices[current.mask_indices_start + i];

        float *sample = samples + i * n_features;
        float *dsample = dataset + index * n_features; 

        for (uint32 f = 0; f < n_features; f++) {
          sample[f] = dsample[f];
        }
        
      }

      uint32 n_splits = 2;  // @TODO: make this a variable number of splits.

      init_centroids_to_first(samples, current.n_samples, n_features, n_splits, centroid_inits);
      k_means(samples, current.n_samples, n_features, n_splits, tolerance, max_iterations, centroid_inits, NULL, clusters);

      update_mask_by_offset(clusters, cluster_id, mask, mask_indices, current.mask_indices_start, current.n_samples);

      uint32 mask_indices_at = current.mask_indices_start;
      for (uint32 offset = 0; offset < n_splits; offset++) {
        ClusterJob child = create_child_clusterjob(offset, clusters, mask_indices_at, mask_indices, &current, n_features);
        queue.push(child);
        
        mask_indices_at += child.n_samples;
      }

      cluster_id += n_splits;

    } else {
      update_mask(cluster_id, mask, mask_indices, current.mask_indices_start, current.n_samples);

      current.layer += 1;
      queue.push(current);

      cluster_id += 1;
    }
  }

  free(mask);
  free(mask_indices);
  free(clusters);
  free(samples);  
  free(centroid_inits);

  *layers_result = layer;
}

void copy_fractal_k_means_result(uint32 n_samples, uint32 *dst) {

  uint32 layer_count = 0;
  while (fractal_result.size() > 0) {
    uint32 *mask = fractal_result.front();
    fractal_result.pop_front();

    uint32 *layer = dst + layer_count * n_samples;
    for (uint32 s = 0; s < n_samples; s++) {
      layer[s] = mask[s];
    }

    layer_count += 1;

    free(mask);

  }
}

