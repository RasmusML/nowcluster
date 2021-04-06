#include <stdio.h>
#include <stdlib.h>
#include <float.h> // FLT_MAX
#include <string.h> // memcpy
#include <assert.h>
#include <queue>
#include <vector>

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
  
  if (group_result != NULL) {  
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
  free(cluster_group_count);
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
  
  ClusterJob root; // @TODO: memory arena
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


/*
struct Cluster {
  
  float *samples;
  uint32 num_samples;
  
  uint32 id;
  uint32 layer;
    
  uint32 *mask_indices;
};

uint32 count(uint32 target, uint32 *arr, uint32 arr_len) {
  uint32 count = 0;
  
  for (uint32 i = 0; i < arr_len; i++) {
    if (arr[i] == target) count += 1;
  }
  
  return count;
}

Cluster *extract(uint32 group, uint32 *child_groups, Cluster *parent, uint32 n_features) {
  uint32 group_size = count(group, child_groups, parent->num_samples);
  
  Cluster *cluster = (Cluster *)malloc(sizeof(Cluster));
  cluster->num_samples = group_size;
  cluster->samples = (float *)malloc(group_size * n_features * sizeof(float));
  cluster->id = 2*parent->id + group;
  cluster->layer = parent->layer + 1;
  cluster->mask_indices = (uint32 *)malloc(group_size * sizeof(uint32)); // @speed
  
  uint32 i = 0;
  for (uint32 j = 0; j < parent->num_samples; j++) {
    if (child_groups[j] == group) {
      cluster->mask_indices[i] = parent->mask_indices[j];
      memcpy(&cluster->samples[i*n_features], &parent->samples[j*n_features], n_features * sizeof(float));
      i += 1;
    }
  }
  
  return cluster;
}

void update_mask(uint32 *mask, uint32 *groups, Cluster *parent) {
  for (int i = 0; i < parent->num_samples; i++) {
    mask[parent->mask_indices[i]] = 2 * parent->id + groups[i];
  }
}

uint32 *copy_mask(uint32 *mask, uint32 length) {
  uint32 *copy = (uint32 *)malloc(length*sizeof(uint32));
  
  for (uint32 i = 0; i < length; i++) {
    copy[i] = mask[i];
  }
  
  return copy;
}

void update_result(uint32 *mask, uint32 *result, uint32 layer, uint32 n_samples) {
  for (uint32 i = 0; i < n_samples; i++) {
    *(result + n_samples * layer + i) = mask[i];
  }
}

//@note max 2^32 clusters
void fractal_k_means_aux(uint32 n_features, uint32 n_samples, Cluster *parent, uint32 *mask, uint32 n_layers, uint32 *result) {
  if (parent->layer == n_layers) return;
  // if (parent->num_samples == 1) return;
  
  uint32 *childrens_group = (uint32 *)malloc(parent->num_samples*sizeof(uint32));
  k_means(parent->samples, parent->num_samples, n_features, 2, 0.001f, 100, NULL, childrens_group);
  
  update_mask(mask, childrens_group, parent);
  
  uint32 *child_mask = copy_mask(mask, n_samples);
  
  Cluster *cluster1 = extract(0, childrens_group, parent, n_features);
  Cluster *cluster2 = extract(1, childrens_group, parent, n_features);
  
  free(parent->samples);
  free(parent->mask_indices);
  
  fractal_k_means_aux(n_features, n_samples, cluster1, child_mask, n_layers, result);
  fractal_k_means_aux(n_features, n_samples, cluster2, child_mask, n_layers, result);
  
  update_result(child_mask, result, parent->layer+1, n_samples);
  free(child_mask);
}


void fractal_k_means(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_layers, uint32 *result) {
  Cluster *root = (Cluster *)malloc(sizeof(Cluster));
  root->num_samples = n_samples;
  root->samples = (float *)malloc(n_samples * n_features * sizeof(float)); // @speed
  memcpy(root->samples, dataset, n_samples * n_features * sizeof(float));
  
  root->id = 0;
  root->layer = 0;
  root->mask_indices = (uint32 *)malloc(n_samples * sizeof(uint32)); // @speed
  
  for (uint32 i = 0; i < n_samples; i++) {
    root->mask_indices[i] = i;
  }
  
  uint32 *mask = (uint32 *)malloc(n_samples * sizeof(uint32));
  
  for (uint32 i = 0; i < n_samples; i++) {
    mask[i] = 0;
  }
  
  fractal_k_means_aux(n_features, n_samples, root, mask, n_layers, result);
  
  update_result(mask, result, 0, n_samples);
  free(mask);
}
*/
