
#include <stdlib.h>

#include "../types.h"

//void k_means(float *dataset, uint32 n_observations, uint32 n_features, uint32 n_clusters, float tolerance, uint32 max_iterations, float *centroid_init, float *centroids_result, uint32 *groups_result);

int main() {
    
    uint32 N = 10_000;
    uint32 M = 4;
    float *dataset = (float *)malloc(N*M*sizeof(float));
    

    //k_means(dataset, N, M, 4, 0.001f, 100, NULL, NULL, NULL);

    return 0;
}