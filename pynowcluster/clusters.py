from .ccore_loader import *
import ctypes as ct
import numpy as np

class Definitions:
  INIT_PROVIDED = 0
  INIT_RANDOMLY = 1
  INIT_KMEANS_PLUS_PLUS = 2
  INIT_TO_FIRST_SAMPLES = 3

class KMeans():
  
  def __init__(self):
    self._centroid_inits = {
      "random" : Definitions.INIT_RANDOMLY,
      "kmeans++" : Definitions.INIT_KMEANS_PLUS_PLUS,
      "first" : Definitions.INIT_TO_FIRST_SAMPLES
    }

    self._ccore = ccore_library.get()
    
    self._ccore.k_means.argtypes = [
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      ct.c_uint32, 
      ct.c_uint32, 
      ct.c_uint32,
      ct.c_float,
      ct.c_uint32,
      ct.c_uint32,
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 

      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"), 
    ]

    self._ccore.k_means.restype = None

  def process(self, X, n_clusters, centroid_init = "kmeans++", tolerance = 0.001, max_iterations = 0):
    """
    @TODO: params description

    centroid_init = {"kmeans++", "random", "first", np.darray}

    """
    self.__check_parameters(X, n_clusters, centroid_init, tolerance, max_iterations)

    if isinstance(centroid_init, np.ndarray):
      centroid_init_type = Definitions.INIT_PROVIDED
    else:
      centroid_init_type = self._centroid_inits[centroid_init]
      centroid_init = np.empty((0,0), dtype=np.float32)

    centroids_result = np.empty((n_clusters, X.shape[1]), dtype=np.float32) 
    clusters_result = np.empty(X.shape[0], dtype=np.uint32)

    self._ccore.k_means(
      X, 
      ct.c_uint32(X.shape[0]), 
      ct.c_uint32(X.shape[1]), 
      ct.c_uint32(n_clusters),
      ct.c_float(tolerance),
      ct.c_uint32(max_iterations),
      ct.c_uint32(centroid_init_type),
      centroid_init,

      centroids_result,
      clusters_result)
  
    self._centroids = centroids_result
    self._clusters = clusters_result
    self._converged = True  # TODO

    return self

  def __check_parameters(self, X, n_clusters, centroid_init, tolerance, max_iterations):
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

    if isinstance(centroid_init, np.ndarray):
      if X.ndim != 2:
         raise TypeError("centroid_init should have 2 dimensions.")

      if X.shape[0] != n_clusters or X.shape[1] != X.shape[1]:
        raise TypeError("centroid_init should have shape:" + n_clusters + " x " + X.shape[1])

    else:
      if self._centroid_inits[centroid_init] is None:
        raise TypeError("invalid centroid init")


class FractalKMeans():

  def __init__(self):
    self._centroid_inits = {
      "random" : Definitions.INIT_RANDOMLY,
      "kmeans++" : Definitions.INIT_KMEANS_PLUS_PLUS,
      "first" : Definitions.INIT_TO_FIRST_SAMPLES
    }

    self._ccore = ccore_library.get()

    self._ccore.fractal_k_means.argtypes = [
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      ct.c_uint32, 
      ct.c_uint32,
      ct.c_uint32,
      ct.c_float,
      ct.c_uint32,
      ct.c_uint32,

      ct.POINTER(ct.c_uint32), 
    ]

    self._ccore.fractal_k_means.restype = None

    self._ccore.copy_fractal_k_means_result.argtypes = [
      ct.c_uint32, 

      np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2, flags="C_CONTIGUOUS")
    ]

    self._ccore.copy_fractal_k_means_result.restype = None

  def process(self, X, min_cluster_size = 1, centroid_init = "kmeans++", tolerance = 0.001, max_iterations = 0):
    self.__check_parameters(X, min_cluster_size, centroid_init, tolerance, max_iterations)

    init_type = self._centroid_inits[centroid_init]
    print("init type: ", init_type)

    layers_result = ct.c_uint32()
    
    self._ccore.fractal_k_means(
        X, 
        ct.c_uint32(X.shape[0]),
        ct.c_uint32(X.shape[1]),
        min_cluster_size,
        tolerance,
        max_iterations,
        init_type,

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

  def __check_parameters(self, X, min_cluster_size, centroid_init, tolerance, max_iterations):

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

    if min_cluster_size < 1:
      raise ValueError("min_cluster_size has to >= 1.")

    if self._centroid_inits[centroid_init] is None:
      raise TypeError("invalid centroid init")

