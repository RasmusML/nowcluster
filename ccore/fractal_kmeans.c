#include <stdlib.h>
#include <string.h> // memcpy, memset

#include "kmeans.h"
#include "kmeans_wcs.h"
#include "fractal_kmeans.h"
#include "arena.h"
#include "ringbuffer.h"
#include "initialization_procedures.h"

struct ClusterJob {
  uint32 layer;
  uint32 n_samples;
  uint32 mask_indices_start;
};

typedef struct ClusterJob ClusterJob;

static
void update_mask(uint32 id, uint32 *mask, uint32 *mask_indices, uint32 mask_indices_start, uint32 n_samples) {
  for (uint32 i = 0; i < n_samples; i++) {
    uint32 mask_index = mask_indices_start + i;
    mask[mask_indices[mask_index]] = id;
  }
}

static
void update_mask_by_offsets(uint32 *offsets, uint32 id, uint32 *mask, uint32 *mask_indices, uint32 mask_indices_start, uint32 n_samples) {
  for (uint32 i = 0; i < n_samples; i++) {
    uint32 mask_index = mask_indices_start + i;
    mask[mask_indices[mask_index]] = id + offsets[i];
  }
}

#define SWAP(x, y, T) do { T _SWAP = x; x = y; y = _SWAP; } while (0)

static
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

Buffer fractal_kmeans_allocate_buffer(uint32 n_samples, uint32 n_features, uint32 n_clusters) {
  size_t mask_size = n_samples * sizeof(uint32);
  size_t mask_indices_size = n_samples * sizeof(uint32);
  size_t clusters_size = n_samples * sizeof(uint32);
  size_t samples_size = n_samples * n_features * sizeof(float);

  size_t centroid_inits_size = n_clusters * n_features * sizeof(float);

  size_t buffer_size = mask_size + mask_indices_size + clusters_size + samples_size + centroid_inits_size + 5 * DEFAULT_ALIGNMENT;
  void *memory = malloc(buffer_size);

  Buffer buffer;
  buffer.size = buffer_size;
  buffer.memory = memory;

  return buffer;
}

Fractal_Kmeans_Result fractal_kmeans(float *dataset, uint32 n_samples, uint32 n_features, uint32 min_cluster_size, 
                                     float tolerance, uint32 max_iterations, uint32 init_method, uint32 split_size, uint8 use_wcss) {  
  
  const uint32 max_centroids = split_size;

  Buffer split_buffer;
  if (use_wcss) {
    split_buffer = kmeans_allocate_buffer(n_samples, n_features, max_centroids);
  } else {
    split_buffer = kmeans_wcs_allocate_buffer(n_samples, n_features, max_centroids);
  }

  size_t mask_size = n_samples * sizeof(uint32);
  size_t mask_indices_size = n_samples * sizeof(uint32);
  size_t clusters_size = n_samples * sizeof(uint32);
  size_t samples_size = n_samples * n_features * sizeof(float);
  size_t centroid_inits_size = max_centroids * n_features * sizeof(float);

  size_t buffer_size = mask_size + mask_indices_size + clusters_size + samples_size + centroid_inits_size + 5 * DEFAULT_ALIGNMENT;

  Buffer buffer = fractal_kmeans_allocate_buffer(n_samples, n_features, max_centroids);

  Arena arena = {0};
  arena_init(&arena, buffer.memory, buffer_size);

  uint32 *mask = (uint32 *) arena_alloc(&arena, mask_size);
  uint32 *mask_indices = (uint32 *) arena_alloc(&arena, mask_indices_size);

  uint32 *clusters = (uint32 *)arena_alloc(&arena, clusters_size);
  float *samples = (float *)arena_alloc(&arena, samples_size);

  float *centroid_inits = (float *)arena_alloc(&arena, centroid_inits_size);

  for (uint32 i = 0; i < n_samples; i++) {
    mask_indices[i] = i;
  }
  
  uint32 layer = 0;
  uint32 mask_id = 0;

  uint8 splitting = 0;
  
  Queue *layers = (Queue *)malloc(sizeof(Queue));
  queue_init(layers);

  const uint32 MAX_JOBS = n_samples + 1;
  RingBuffer jobs;
  ringbuffer_init(MAX_JOBS, sizeof(ClusterJob), &jobs);

  ClusterJob *root = (ClusterJob *)ringbuffer_alloc(&jobs);
  root->n_samples = n_samples;
  root->layer = 0;
  root->mask_indices_start = 0;

  uint32 converged_result = 1;

  while (1) {
    ClusterJob *current = (ClusterJob *)ringbuffer_get_first(&jobs);
    
    if (current->layer > layer) {
      
      if (splitting == 0) break;
      splitting = 0;

      mask_id = 0;
      layer += 1;

      uint32 *mask_result = (uint32 *)malloc(n_samples * sizeof(uint32));
      memcpy(mask_result, mask, mask_size);

      queue_enqueue((void *)mask_result, layers);
    }
    
    if (current->n_samples > min_cluster_size) {
      splitting = 1;

      for (uint32 i = 0; i < current->n_samples; i++) {
        uint32 index = mask_indices[current->mask_indices_start + i];

        float *sample = samples + i * n_features;
        float *dataset_sample = dataset + index * n_features; 

        for (uint32 f = 0; f < n_features; f++) {
          sample[f] = dataset_sample[f];
        }
      }


      uint32 n_splits = split_size;
      if (n_splits > current->n_samples) n_splits = current->n_samples;

      init_centroids(init_method, samples, current->n_samples, n_features, n_splits, centroid_inits, NULL);
      
      uint32 converged;

      if (use_wcss) kmeans_algorithm(samples, current->n_samples, n_features, n_splits, tolerance, max_iterations, centroid_inits, NULL, clusters, &converged, &split_buffer);
      else kmeans_wcs_algorithm(samples, current->n_samples, n_features, n_splits, tolerance, max_iterations, centroid_inits, NULL, clusters, &converged, &split_buffer);
      
      if (converged == 0) converged_result = 0;

      update_mask_by_offsets(clusters, mask_id, mask, mask_indices, current->mask_indices_start, current->n_samples);

      uint32 mask_indices_at = current->mask_indices_start;
      for (uint32 offset = 0; offset < n_splits; offset++) {
        uint32 cluster_size = update_mask_indices(offset, clusters, mask_indices_at, mask_indices, current->mask_indices_start, current->n_samples);

        ClusterJob *child = (ClusterJob *)ringbuffer_alloc(&jobs);
        child->n_samples = cluster_size;
        child->layer = current->layer + 1;
        child->mask_indices_start = mask_indices_at;

        mask_indices_at += cluster_size;
      }

      mask_id += n_splits;

    } else {
      update_mask(mask_id, mask, mask_indices, current->mask_indices_start, current->n_samples);

      ClusterJob *copy = (ClusterJob *)ringbuffer_alloc(&jobs);
      copy->layer = layer + 1;
      copy->n_samples = current->n_samples;
      copy->mask_indices_start = current->mask_indices_start;

      mask_id += 1;
    }

    ringbuffer_free(&jobs);

  }

  arena_free_all(&arena);
  free(buffer.memory);

  Fractal_Kmeans_Result result;
  result.layers = layers;
  result.num_layers = layer;
  result.converged = converged_result;

  return result;
}

void consume_fractal_kmeans_result_into_array(uint32 n_samples, uint32 *array, Fractal_Kmeans_Result *result) {
  size_t mask_size = n_samples * sizeof(uint32);

  for (uint32 i = 0; i < result->num_layers; i++) {
    uint32 *mask = (uint32 *)queue_dequeue(result->layers);
    uint32 *layer = array + i * n_samples;
    memcpy(layer, mask, mask_size);
    
    free(mask);
  }

  queue_free(result->layers);
  free(result);
}