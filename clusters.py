import ctypes as ct
import numpy as np

# @TODO: consider implementing the c-extension module instead of using ctypes, it may yield better startup-performance. Profile c-extension module and ctypes.

lib = np.ctypeslib.load_library("kmeans", "build")

def init():    
    k_means_func = lib.k_means
    k_means_func.argtypes = [
          np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
          ct.c_uint32, 
          ct.c_uint32, 
          ct.c_uint32,
          ct.c_float,
          ct.c_uint32,
          np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
          np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"), 
        ]

    k_means_func.restype = None

def k_means(X, num_clusters, tolerance = 0.01, max_iterations = 0):
    centroids_result = np.empty((num_clusters, X.shape[1]), np.float32) 
    cluster_groups_result = np.empty(X.shape[0], dtype=np.uint32)

    lib.k_means(
        X, 
        ct.c_uint32(X.shape[0]), 
        ct.c_uint32(X.shape[1]), 
        ct.c_uint32(num_clusters),
        ct.c_float(tolerance),
        ct.c_uint32(max_iterations),
        centroids_result,
        cluster_groups_result)

    return centroids_result, cluster_groups_result

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    init()
    
    clusters = 4

    n = 1000
    x = np.random.normal(0, 10, n);
    y = np.random.normal(0, 5, n);

    X = np.stack((x,y), axis=1)
    X = X.astype(np.float32);

    centroids, groups = k_means(X, clusters)
    #print(centroids, groups)

    #deinit()

    for i in range(clusters):
        points = X[groups == i] 
        plt.scatter(points[:,0], points[:,1])

    plt.scatter(centroids[:,0], centroids[:,1], color="black")
    plt.show()
    

