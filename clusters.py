import ctypes as ct
import numpy as np

# @TODO: consider implementing the c-extension module instead of using ctypes, it may yield better startup-performance. Profile c-extension module and ctypes.

# @TODO: make this into a class

lib = np.ctypeslib.load_library("k_means", "build")
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

# @TODO: check for valid parameters
def k_means(X, n_clusters, tolerance = 0.01, max_iterations = 0):
    centroids_result = np.empty((n_clusters, X.shape[1]), dtype=np.float32) 
    cluster_groups_result = np.empty(X.shape[0], dtype=np.uint32)

    lib.k_means(
        X, 
        ct.c_uint32(X.shape[0]), 
        ct.c_uint32(X.shape[1]), 
        ct.c_uint32(n_clusters),
        ct.c_float(tolerance),
        ct.c_uint32(max_iterations),
        centroids_result,
        cluster_groups_result)

    return centroids_result, cluster_groups_result
