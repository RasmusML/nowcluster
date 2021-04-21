#include <list>

#include "clusters.h"

void k_means(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float tolerance, uint32 max_iterations, uint32 init_method, float *custom_centroid_init, float *centroids_result, uint32 *groups_result) {
  k_means_algorithm(dataset, n_samples, n_features, n_clusters, tolerance, max_iterations, init_method, custom_centroid_init, centroids_result, groups_result);
}

std::list<uint32 *> fractal_result;

void fractal_k_means(float *dataset, uint32 n_samples, uint32 n_features, uint32 min_cluster_size, 
                          float tolerance, uint32 max_iterations, uint32 init_method, uint32 *layers_result) {  

  fractal_k_means_full(dataset, n_samples, n_features, min_cluster_size, tolerance, max_iterations, init_method, layers_result, fractal_result);
}

void copy_fractal_k_means_result(uint32 n_samples, uint32 *dst) {
  copy_fractal_k_means_result_full(n_samples, dst, fractal_result);
}