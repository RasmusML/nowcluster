#ifndef KMEANS_WCS_H
#define KMEANS_WCS_H

#include "types.h"
#include "buffer.h"

void kmeans_wcs_algorithm(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, 
                          float tolerance, uint32 max_iterations, float *centroid_init, float *centroids_result, 
                          uint32 *groups_result, uint32 *converged_result, Buffer *buffer);

Buffer kmeans_wcs_allocate_buffer(uint32 n_samples, uint32 n_features, uint32 n_clusters);

#endif