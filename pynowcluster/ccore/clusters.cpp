#include <stdio.h>
#include <stdlib.h>
#include <float.h> // DBL_MAX
#include <string.h> // memcpy, memset
#include <assert.h>
//#include <vector>
#include <queue>
#include <list>
#include <thread>

#include "initialization_procedures.h"
#include "timer.h"
#include "memory.h"


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

//
// K-means
//

inline double squared_euclidian_distance(float *v1, float *v2, uint32 n_elements) {
  double dst = 0;
  for (uint32 i = 0; i < n_elements; i++) {
    float dt = v1[i] - v2[i];
    dst += dt * dt;
  }
  return dst;
}

static int8 kmeans_init_centroids(float *dataset, uint32 n_samples, uint32 n_features, 
                           uint32 n_clusters, uint32 init_method, float *custom_centroid_init, float *centroid_init) {
  if (init_method == INIT_PROVIDED)
    memcpy(centroid_init, custom_centroid_init, n_clusters * n_features * sizeof(float));
  else if (init_method == INIT_KMEANS_PLUS_PLUS)
    init_centroids_using_kmeansplusplus(dataset, n_samples, n_features, n_clusters, centroid_init);
  else if (init_method == INIT_RANDOMLY)
    init_centroids_randomly(dataset, n_samples, n_features, n_clusters, centroid_init);     
  else if (init_method == INIT_TO_FIRST_SAMPLES)
    init_centroids_to_first_samples(dataset, n_samples, n_features, n_clusters, centroid_init);     
  else
    return -1;

  return 0;
}

void k_means_algorithm_full(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, 
                            float tolerance, uint32 max_iterations, float *centroid_init, 
                            float *centroids_result, uint32 *groups_result, uint32 *converged_result) {
               
  size_t centroids_size = n_clusters * n_features * sizeof(float);
  size_t cluster_sizes_size = n_clusters * sizeof(uint32);
  size_t clusters_size = n_samples * sizeof(uint32);
  size_t new_centroids_size = centroids_size;

  // :difference
  //size_t lambda_inv_size = n_clusters * sizeof(float);

  float *centroids;
  uint32 *cluster_sizes;
  uint32 *clusters;
  float *new_centroids;

  centroids = (float *)malloc(centroids_size);
  cluster_sizes = (uint32 *)malloc(cluster_sizes_size);
  clusters = (uint32 *)malloc(clusters_size);
  new_centroids = (float *)malloc(new_centroids_size); 

  memcpy(centroids, centroid_init, centroids_size);

  // :difference
  //float *lambda_inv = (float *)malloc(lambda_inv_size);

  uint32 converged = 1;

  uint32 iteration = 1;
  while(1) {
    // reset new centroids + cluster sizes
    memset(cluster_sizes, 0, cluster_sizes_size);
    memset(new_centroids, 0, new_centroids_size);

    // :difference
    //memset(lambda_inv, 0, lambda_size);

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
        new_centroid[f] += sample[f];

        // :difference
        //new_centroid[f] += sample[f] / min_distance;
        //lambda_inv[f] += 1 / min_distance;
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
        new_centroid[f] /= group_count;

        // :difference
        //new_centroid[f] /= lambda_inv[f];
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

  free(centroids);
  free(cluster_sizes);
  free(clusters);
  free(new_centroids);

}


void k_means_algorithm(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, 
                       float tolerance, uint32 max_iterations, uint32 init_method, float *custom_centroid_init, 
                       float *centroids_result, uint32 *groups_result, uint32 *converged_result) {

  float centroids_size = n_clusters * n_features * sizeof(float);
  float *centroid_init = (float *)malloc(centroids_size);
  
  int8 init_success = kmeans_init_centroids(dataset, n_samples, n_features, n_clusters, init_method, custom_centroid_init, centroid_init);
  assert(init_success == 0);
  
  k_means_algorithm_full(dataset, n_samples, n_features, n_clusters, tolerance, max_iterations, centroid_init, centroids_result, groups_result, converged_result);
  free(centroid_init);
}

//
// Fractal K-means
//

struct ClusterJob {
  uint32 layer;
  uint32 n_samples;
  uint32 mask_indices_start;
  // uint32 sample_pointer;
};

void update_mask(uint32 id, uint32 *mask, uint32 *mask_indices, uint32 mask_indices_start, uint32 n_samples) {
  for (uint32 i = 0; i < n_samples; i++) {
    uint32 mask_index = mask_indices_start + i;
    mask[mask_indices[mask_index]] = id;
  }
}

void update_mask_by_offsets(uint32 *offsets, uint32 id, uint32 *mask, uint32 *mask_indices, uint32 mask_indices_start, uint32 n_samples) {
  for (uint32 i = 0; i < n_samples; i++) {
    uint32 mask_index = mask_indices_start + i;
    mask[mask_indices[mask_index]] = id + offsets[i];
  }
}

#define SWAP(x, y, T) do { T _SWAP = x; x = y; y = _SWAP; } while (0)

uint32 update_mask_indices(uint32 cluster, uint32 *clusters, uint32 mask_indices_at, uint32* mask_indices, uint32 mask_indices_start, uint32 n_samples) {
  uint32 offset = mask_indices_at - mask_indices_start;
  uint32 i = offset;

  for (uint32 j = offset; j < n_samples; j++) {

    if (clusters[j] == cluster) {
      SWAP(clusters[i], clusters[j], uint32);
      SWAP(mask_indices[mask_indices_start + i], mask_indices[mask_indices_start + j], uint32);
      
      i += 1;
    }
  }

  return i - offset;
}

static int8 fractal_init_centroids(float *dataset, uint32 n_samples, uint32 n_features, 
                                   uint32 n_clusters, uint32 init_method, float *centroid_init) {
                                
  if (init_method == INIT_KMEANS_PLUS_PLUS)
    init_centroids_using_kmeansplusplus(dataset, n_samples, n_features, n_clusters, centroid_init);
  else if (init_method == INIT_RANDOMLY)
    init_centroids_randomly(dataset, n_samples, n_features, n_clusters, centroid_init);     
  else if (init_method == INIT_TO_FIRST_SAMPLES)
    init_centroids_to_first_samples(dataset, n_samples, n_features, n_clusters, centroid_init);     
  else
    return -1;

  return 0;
}

void fractal_k_means_full(float *dataset, uint32 n_samples, uint32 n_features, uint32 min_cluster_size, 
                          float tolerance, uint32 max_iterations, uint32 init_method, 
                          std::list<uint32 *> &fractal_result, uint32 *converged_result) {  
  
  const uint32 max_centroids = 2;

  const uint32 processor_count = std::thread::hardware_concurrency();
  printf("logical processors %u\n", processor_count);

  // fractal memory
  size_t mask_size = n_samples * sizeof(uint32);
  size_t mask_indices_size = n_samples * sizeof(uint32);
  size_t clusters_size = n_samples * sizeof(uint32);
  size_t samples_size = n_samples * n_features * sizeof(float);

  size_t centroid_inits_size = max_centroids * n_features * sizeof(float);

  uint32 *mask = (uint32 *) malloc(mask_size);
  uint32 *mask_indices = (uint32 *) malloc(mask_indices_size);

  for (uint32 i = 0; i < n_samples; i++) {
    mask_indices[i] = i;
  }

  uint32 *clusters = (uint32 *)malloc(clusters_size);
  float *samples = (float *)malloc(samples_size);

  float *centroid_inits = (float *)malloc(centroid_inits_size);

  ClusterJob root;
  root.n_samples = n_samples;
  root.layer = 0;
  root.mask_indices_start = 0;
  
  uint32 layer = 0;
  uint32 mask_id = 0;

/*
  uint32 sample_index = 0;
  uint32 clusterjobs_left_in_current_layer = 1;
  uint32 clusterjobs_in_next_layer = 0;
*/  

  uint8 splitting = 0;
  
  std::queue<ClusterJob> queue;
  
  queue.push(root);

  *converged_result = 1;

  while (queue.size() > 0) {
    ClusterJob current = queue.front();
    queue.pop();
    
    if (current.layer > layer) {
      
      if (splitting == 0) break;
      splitting = 0;

      mask_id = 0;
      layer += 1;

      uint32 *mask_result = (uint32 *)malloc(n_samples * sizeof(uint32));
      memcpy(mask_result, mask, mask_size);

      fractal_result.push_back(mask_result);
    }
    
    // @TODO: figure out what to do if zero points are assigned to a cluster. probably just return K-1 clusters, and say only K-1 clusters exist.
    // That should fix the problem were a cluster contains 0 centroids and fractal k-means keeps creating layers
    if (current.n_samples > min_cluster_size) {
      splitting = 1;

      for (uint32 i = 0; i < current.n_samples; i++) {
        uint32 index = mask_indices[current.mask_indices_start + i];

        float *sample = samples + i * n_features;
        float *dataset_sample = dataset + index * n_features; 

        for (uint32 f = 0; f < n_features; f++) {
          sample[f] = dataset_sample[f];
        }
      }

      uint32 n_splits = 2;  // @TODO: make this a variable number of splits. It has to take min_cluster_size into account though

      uint32 converged;
      fractal_init_centroids(samples, current.n_samples, n_features, n_splits, init_method, centroid_inits);
      k_means_algorithm_full(samples, current.n_samples, n_features, n_splits, tolerance, 
                             max_iterations, centroid_inits, NULL, clusters, &converged);
      if (converged == 0) *converged_result = 0;

      update_mask_by_offsets(clusters, mask_id, mask, mask_indices, current.mask_indices_start, current.n_samples);

      uint32 mask_indices_at = current.mask_indices_start;
      for (uint32 offset = 0; offset < n_splits; offset++) {
        uint32 cluster_size = update_mask_indices(offset, clusters, mask_indices_at, mask_indices, current.mask_indices_start, current.n_samples);

        ClusterJob child;
        child.n_samples = cluster_size;
        child.layer = current.layer + 1;
        child.mask_indices_start = mask_indices_at;

        queue.push(child);
        
        mask_indices_at += cluster_size;
      }

      mask_id += n_splits;

    } else {
      update_mask(mask_id, mask, mask_indices, current.mask_indices_start, current.n_samples);

      current.layer += 1;
      queue.push(current);

      mask_id += 1;
    }
  }

  free(mask);
  free(mask_indices);
  free(samples);
  free(centroid_inits);
}

void fractal_k_means_parallel_full(float *dataset, uint32 n_samples, uint32 n_features, uint32 min_cluster_size, 
                                   float tolerance, uint32 max_iterations, uint32 init_method, 
                                   std::list<uint32 *> &fractal_result, uint32 *converged_result) {  
  
  const uint32 max_centroids = 2;

  const uint32 processor_count = std::thread::hardware_concurrency();
  printf("logical processors %u\n", processor_count);

  // fractal memory
  size_t mask_size = n_samples * sizeof(uint32);
  size_t mask_indices_size = n_samples * sizeof(uint32);
  size_t clusters_size = n_samples * sizeof(uint32);
  size_t samples_size = n_samples * n_features * sizeof(float);

  size_t centroid_inits_size = max_centroids * n_features * sizeof(float);

  uint32 *mask = (uint32 *) malloc(mask_size);
  uint32 *mask_indices = (uint32 *) malloc(mask_indices_size);

  for (uint32 i = 0; i < n_samples; i++) {
    mask_indices[i] = i;
  }

  uint32 *clusters = (uint32 *)malloc(clusters_size);
  float *samples = (float *)malloc(samples_size);

  float *centroid_inits = (float *)malloc(centroid_inits_size);

  ClusterJob root;
  root.n_samples = n_samples;
  root.layer = 0;
  root.mask_indices_start = 0;
  
  uint32 layer = 0;
  uint32 mask_id = 0;

/*
  uint32 sample_index = 0;
  uint32 clusterjobs_left_in_current_layer = 1;
  uint32 clusterjobs_in_next_layer = 0;
*/  

  uint8 splitting = 0;
  
  std::queue<ClusterJob> queue;
  
  queue.push(root);

  *converged_result = 1;

  while (queue.size() > 0) {
    ClusterJob current = queue.front();
    queue.pop();
    
    if (current.layer > layer) {
      
      if (splitting == 0) break;
      splitting = 0;

      mask_id = 0;
      layer += 1;

      uint32 *mask_result = (uint32 *)malloc(n_samples * sizeof(uint32));
      memcpy(mask_result, mask, mask_size);

      fractal_result.push_back(mask_result);
    }
    
    
  }

  free(mask);
  free(mask_indices);
  free(samples);
  free(centroid_inits);
}

void process_clusterjob(ClusterJob job) {
  // @TODO: figure out what to do if zero points are assigned to a cluster. probably just return K-1 clusters, and say only K-1 clusters exist.
  // That should fix the problem were a cluster contains 0 centroids and fractal k-means keeps creating layers
  if (current.n_samples > min_cluster_size) {
    splitting = 1;

    for (uint32 i = 0; i < current.n_samples; i++) {
      uint32 index = mask_indices[current.mask_indices_start + i];

      float *sample = samples + i * n_features;
      float *dataset_sample = dataset + index * n_features; 

      for (uint32 f = 0; f < n_features; f++) {
        sample[f] = dataset_sample[f];
      }
    }

    uint32 n_splits = 2;  // @TODO: make this a variable number of splits. It has to take min_cluster_size into account though

    uint32 converged;
    fractal_init_centroids(samples, current.n_samples, n_features, n_splits, init_method, centroid_inits);
    k_means_algorithm_full(samples, current.n_samples, n_features, n_splits, tolerance, 
                            max_iterations, centroid_inits, NULL, clusters, &converged);
    if (converged == 0) *converged_result = 0;

    update_mask_by_offsets(clusters, mask_id, mask, mask_indices, current.mask_indices_start, current.n_samples);

    uint32 mask_indices_at = current.mask_indices_start;
    for (uint32 offset = 0; offset < n_splits; offset++) {
      uint32 cluster_size = update_mask_indices(offset, clusters, mask_indices_at, mask_indices, current.mask_indices_start, current.n_samples);

      ClusterJob child;
      child.n_samples = cluster_size;
      child.layer = current.layer + 1;
      child.mask_indices_start = mask_indices_at;

      queue.push(child);
      
      mask_indices_at += cluster_size;
    }

    mask_id += n_splits;

  } else {
    update_mask(mask_id, mask, mask_indices, current.mask_indices_start, current.n_samples);

    current.layer += 1;
    queue.push(current);

    mask_id += 1;
  }
}

void copy_fractal_k_means_layer_queue_into_array(uint32 n_samples, uint32 *dst, std::list<uint32 *> &fractal_result) {
  size_t mask_size = n_samples * sizeof(uint32);

  uint32 layer_count = 0;
  while (fractal_result.size() > 0) {
    uint32 *mask = fractal_result.front();
    fractal_result.pop_front();

    uint32 *layer = dst + layer_count * n_samples;
    memcpy(layer, mask, mask_size);

    layer_count += 1;

    free(mask);
  }
}