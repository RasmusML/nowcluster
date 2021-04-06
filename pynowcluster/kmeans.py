from .ccore_loader import *
import ctypes as ct
import numpy as np
import math # log2 @hack


# @TODO: check for valid parameters
def k_means(X, n_clusters, tolerance = 0.01, max_iterations = 0):
    ccore = ccore_library.get()

    ccore.k_means.argtypes = [
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      ct.c_uint32, 
      ct.c_uint32, 
      ct.c_uint32,
      ct.c_float,
      ct.c_uint32,
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"), 
    ]

    ccore.k_means.restype = None

    centroids_result = np.empty((n_clusters, X.shape[1]), dtype=np.float32) 
    cluster_groups_result = np.empty(X.shape[0], dtype=np.uint32)

    ccore.k_means(
        X, 
        ct.c_uint32(X.shape[0]), 
        ct.c_uint32(X.shape[1]), 
        ct.c_uint32(n_clusters),
        ct.c_float(tolerance),
        ct.c_uint32(max_iterations),
        centroids_result,
        cluster_groups_result)

    return centroids_result, cluster_groups_result


def fractal_k_means(X, n_layers):
    # n_clusters, n_layers?, init: {‘k-means++’, ‘random’}
    ccore = ccore_library.get()

    ccore.fractal_k_means.argtypes = [
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      ct.c_uint32, 
      ct.c_uint32, 
      ct.c_uint32,
      np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2, flags="C_CONTIGUOUS"), 
    ]

    ccore.k_means.restype = None

    #centroids_result = np.empty((n_clusters, X.shape[1]), dtype=np.float32) 
    cluster_groups_result = np.empty((n_layers, X.shape[0]), dtype=np.uint32)

    ccore.fractal_k_means(
        X, 
        ct.c_uint32(X.shape[0]), 
        ct.c_uint32(X.shape[1]), 
        ct.c_uint32(n_layers),
        cluster_groups_result)

    return cluster_groups_result