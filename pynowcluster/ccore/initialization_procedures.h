#ifndef __initialization_procedures_h
#define __initialization_procedures_h

#include <float.h> // DBL_MAX

#include "types.h"

const uint32 INIT_PROVIDED = 0;
const uint32 INIT_RANDOMLY = 1;
const uint32 INIT_KMEANS_PLUS_PLUS = 2;
const uint32 INIT_TO_FIRST_SAMPLES = 3;

void init_centroids_randomly(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids);
void init_centroids_using_kmeansplusplus(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids);
void init_centroids_to_first_samples(float *dataset, uint32 n_samples, uint32 n_features, uint32 n_clusters, float *centroids);


#endif