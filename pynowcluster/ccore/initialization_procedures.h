#pragma once

#include "types.h"

void init_centroids(uint32 init_method, float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroid_init, float *custom_centroid_init);

void init_centroids_randomly(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids);
void init_centroids_using_kmeansplusplus(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids);
void init_centroids_to_first_samples(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids);

