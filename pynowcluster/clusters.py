from .ccore_loader import *
import ctypes as ct
import numpy as np
import math # log2 @hack


class KMeans():
  
  def __init__(self):
    self._ccore = ccore_library.get()
    
    self._ccore.k_means.argtypes = [
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      ct.c_uint32, 
      ct.c_uint32, 
      ct.c_uint32,
      ct.c_float,
      ct.c_uint32,
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"), 
    ]

    self._ccore.k_means.restype = None


  def process(self, X, n_clusters, centroid_init, tolerance = 0.001, max_iterations = 0):
    if not isinstance(X, np.ndarray):
      raise TypeError("X should be a numpy array.")

    if X.ndim != 2:
      raise TypeError("X should have 2 dimensions.")

    if X.dtype != np.float32:
      raise TypeError("X should contain elements of type np.float32.")

    centroids_result = np.empty((n_clusters, X.shape[1]), dtype=np.float32) 
    cluster_groups_result = np.empty(X.shape[0], dtype=np.uint32)

    self._ccore.k_means(
      X, 
      ct.c_uint32(X.shape[0]), 
      ct.c_uint32(X.shape[1]), 
      ct.c_uint32(n_clusters),
      ct.c_float(tolerance),
      ct.c_uint32(max_iterations),
      centroid_init,

      centroids_result,
      cluster_groups_result)
  
    self._centroids = centroids_result
    self._clusters = cluster_groups_result
    self._converged = True  # TODO

    return self



"""
class FractalKMeans():

  def __init__(self):
    self.ccore = ccore_library.get()

  def process(self, X, n_layers):
    # n_clusters, n_layers?, init: {‘k-means++’, ‘random’}

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

  def get_layer(n):
    pass

  def get_n_clusters(n):
    pass

  def get_clusters_with_max_n_samples(n):
    pass
"""