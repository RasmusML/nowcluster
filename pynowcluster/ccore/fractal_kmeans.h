#ifndef _clusters__h
#define _clusters__h

#include <list>

#include "types.h"

void fractal_kmeans(float *dataset, uint32 n_samples, uint32 n_features, uint32 min_cluster_size, 
                    float tolerance, uint32 max_iterations, uint32 init_method, const bool use_wcss, std::list<uint32 *> &fractal_result, uint32 *converged_result);  


void copy_fractal_kmeans_layer_queue_into_array(uint32 n_samples, uint32 *dst, std::list<uint32 *>& fractal_result);

#endif