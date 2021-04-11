from .ccore_loader import *
import ctypes as ct
import numpy as np


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

    if tolerance < 0:
      raise ValueError("tolerance has to be greater than 0")

    if max_iterations < 0:
      raise ValueError("max_iterations has to be non-negative")

    centroids_result = np.empty((n_clusters, X.shape[1]), dtype=np.float32) 
    clusters_result = np.empty(X.shape[0], dtype=np.uint32)

    self._ccore.k_means(
      X, 
      ct.c_uint32(X.shape[0]), 
      ct.c_uint32(X.shape[1]), 
      ct.c_uint32(n_clusters),
      ct.c_float(tolerance),
      ct.c_uint32(max_iterations),
      centroid_init,

      centroids_result,
      clusters_result)
  
    self._centroids = centroids_result
    self._clusters = clusters_result
    self._converged = True  # TODO

    return self


class FractalKMeans():

  def __init__(self):
    self._ccore = ccore_library.get()

    self._ccore.fractal_k_means.argtypes = [
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      ct.c_uint32, 
      ct.c_uint32,
      ct.c_float,
      ct.c_uint32,
      
      ct.POINTER(ct.c_uint32), 
    ]

    self._ccore.fractal_k_means.restype = None

    self._ccore.copy_fractal_k_means_result.argtypes = [
      ct.c_uint32, 

      np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2, flags="C_CONTIGUOUS")
    ]

    self._ccore.copy_fractal_k_means_result.restype = None

  def process(self, X, tolerance = 0.001, max_iterations = 0):
    # @TODO: max_cluster_size, n_layers, init: {‘k-means++’, ‘random’}

    layers_result = ct.c_uint32()
    
    self._ccore.fractal_k_means(
        X, 
        ct.c_uint32(X.shape[0]),
        ct.c_uint32(X.shape[1]),
        tolerance,
        max_iterations,

        ct.byref(layers_result))

    clusters_result = np.empty((layers_result.value, X.shape[0]), dtype=np.uint32) 

    self._ccore.copy_fractal_k_means_result(
      ct.c_uint32(X.shape[0]),

      clusters_result
    )

    self._clusters = clusters_result

    return self

  def get_num_layers(self):
    return self._clusters.shape[0]

  def get_layer(self, n):
    return self._clusters[n,:]

  def get_n_clusters(self, n):
    pass

  def get_clusters_with_max_n_samples(self, n):
    pass
