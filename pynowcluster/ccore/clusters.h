#ifndef _clusters__h
#define _clusters__h

#include <list>

#include "types.h"

void k_means_algorithm(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, 
                       float tolerance, uint32 max_iterations, uint32 init_method, float *custom_centroid_init, 
                       float *centroids_result, uint32 *groups_result);

void k_means_algorithm_full(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, 
                            float tolerance, uint32 max_iterations, float *centroid_init, 
                            float *centroids_result, uint32 *groups_result);

void fractal_k_means_full(float *dataset, uint32 n_samples, uint32 n_features, uint32 min_cluster_size, 
                          float tolerance, uint32 max_iterations, uint32 init_method, uint32 *layers_result, std::list<uint32 *>& fractal_result);  


void copy_fractal_k_means_result_full(uint32 n_samples, uint32 *dst, std::list<uint32 *>& fractal_result);

#endif