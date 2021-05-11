#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "queue.h"
#include "types.h"
#include "kmeans.h"
#include "kmeans_wcs.h"
#include "fractal_kmeans.h"
#include "initialization_procedures.h"

void interface_kmeans(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float tolerance, 
                      uint32 max_iterations, uint32 init_method, float *custom_centroid_init, uint32 use_wcss, 
                      float *centroids_result, uint32 *groups_result, uint32 *converged_result) {

  float centroids_size = n_clusters * n_features * sizeof(float);
  float *centroid_init = (float *)malloc(centroids_size);
  
  init_centroids(init_method, dataset, n_samples, n_features, n_clusters, centroid_init, custom_centroid_init);

  if (use_wcss) {
    Buffer kmeans_buffer = kmeans_allocate_buffer(n_samples, n_features, n_clusters);
    kmeans_algorithm(dataset, n_samples, n_features, n_clusters, tolerance, max_iterations, centroid_init, centroids_result, groups_result, converged_result, &kmeans_buffer);
    free(kmeans_buffer.memory);
  } else {
    Buffer kmeans_wcs_buffer = kmeans_wcs_allocate_buffer(n_samples, n_features, n_clusters);
    kmeans_wcs_algorithm(dataset, n_samples, n_features, n_clusters, tolerance, max_iterations, centroid_init, centroids_result, groups_result, converged_result, &kmeans_wcs_buffer);
    free(kmeans_wcs_buffer.memory);
  }

  free(centroid_init);
}

static Fractal_Kmeans_Result *fractal_result;

void interface_fractal_kmeans(float *dataset, uint32 n_samples, uint32 n_features, uint32 min_cluster_size, float tolerance, 
                              uint32 max_iterations, uint32 init_method, uint32 split_size, uint32 use_wcss, 
                              uint32 *layers_result, uint32 *converged_result) {  

  Fractal_Kmeans_Result result = fractal_kmeans(dataset, n_samples, n_features, min_cluster_size, tolerance, max_iterations, init_method, split_size, use_wcss);
  
  fractal_result = (Fractal_Kmeans_Result *)malloc(sizeof(Fractal_Kmeans_Result));
  memcpy(fractal_result, &result, sizeof(Fractal_Kmeans_Result));

  *layers_result = result.num_layers;
  *converged_result = result.converged;
}

void interface_copy_fractal_kmeans_result(uint32 n_samples, uint32 *dst) {
  transform_fractal_kmeans_result_into_array(n_samples, dst, fractal_result);
  fractal_result = NULL;
}