#ifndef FRACTAL_KMEANS_H
#define FRACTAL_KMEANS_H

#include "queue.h"
#include "types.h"

typedef struct Fractal_Kmeans_Result Fractal_Kmeans_Result;
struct Fractal_Kmeans_Result {
  Queue *layers;
  uint32 num_layers;
  uint32 converged;
};

Fractal_Kmeans_Result fractal_kmeans(float *dataset, uint32 n_samples, uint32 n_features, uint32 min_cluster_size, 
                                     float tolerance, uint32 max_iterations, uint32 init_method, uint32 split_size, uint8 use_wcss);  


void transform_fractal_kmeans_result_into_array(uint32 n_samples, uint32 *array, Fractal_Kmeans_Result *fractal_result);

#endif