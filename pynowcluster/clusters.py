from .ccore_loader import *
import ctypes as ct
import numpy as np

class Definitions:
  INIT_PROVIDED = 0
  INIT_RANDOMLY = 1
  INIT_KMEANS_PLUS_PLUS = 2
  INIT_TO_FIRST_SAMPLES = 3

class KMeans():
  """K-means clustering.

  A wrapper to a C implementation of Lloyd's algorithm. 

  Attributes
  ----------
  centroids : ndarray of shape (n_centroids, n_features) of np.float32
              Cluster centers.

  clusters  : cluster label for each sample index describing which cluster the sample is in.
              Label n corresponds to the centroid in position n in centroids.

  converged : whether KMeans converged or not

  Usage
  -----
  from pynowcluster.clusters import KMeans

  data_size = 100
  num_iters = 50
  num_clusters = 4

  # sample from Gaussians 
  data1 = np.random.normal((5,5), (4, 4), (data_size,2))
  data2 = np.random.normal((4,60), (3,3), (data_size, 2))

  # Combine the data to create the final dataset
  X = np.concatenate((data1,data2), axis = 0)
  X = X.astype(np.float32)

  np.random.shuffle(X)

  kMeans = KMeans().process(X, num_clusters)

  print(kMeans.centroids)
  print(kMeans.converged)
  print(kMeans.clusters)

  """
  
  def __init__(self):
    self._centroid_inits = {
      "random" : Definitions.INIT_RANDOMLY,
      "kmeans++" : Definitions.INIT_KMEANS_PLUS_PLUS,
      "first" : Definitions.INIT_TO_FIRST_SAMPLES
    }

    self._ccore = ccore_library.get()
    
    self._ccore.interface_kmeans.argtypes = [
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      ct.c_uint32, 
      ct.c_uint32, 
      ct.c_uint32,
      ct.c_float,
      ct.c_uint32,
      ct.c_uint32,
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      ct.c_uint32,

      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
      ct.POINTER(ct.c_uint32)
    ]

    self._ccore.interface_kmeans.restype = None

  def process(self, X, n_clusters, centroid_init = "kmeans++", objective_function = "wcss", tolerance = 0.001, max_iterations = 0):
    """
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) of np.float32
        The input samples.

    n_clusters : number of clusters

    centroid_init : {"kmeans++", "random", "first"} or ndarray of shape (n_clusters, n_features)

    objective_function : {"wcss", "wcs"}
                         Whether to minimize the within-cluster sum of squares (wcss) or the within-cluster sum (wcs).

    tolerance : if all centroids move less than tolerance during an update, then the centroids have converged

    max_iterations : maximum number of iterations (0 means iterate till convergens)

    """
    self.__check_parameters(X, n_clusters, centroid_init, objective_function, tolerance, max_iterations)

    if isinstance(centroid_init, np.ndarray):
      centroid_init_type = Definitions.INIT_PROVIDED
    else:
      centroid_init_type = self._centroid_inits[centroid_init]
      centroid_init = np.empty((0,0), dtype=np.float32)

    if objective_function == "wcss":
      objective_function_type = 1
    else:
      objective_function_type = 0

    centroids_result = np.empty((n_clusters, X.shape[1]), dtype=np.float32) 
    clusters_result = np.empty(X.shape[0], dtype=np.uint32)

    converged_result = ct.c_uint32()

    self._ccore.interface_kmeans(
      X, 
      ct.c_uint32(X.shape[0]), 
      ct.c_uint32(X.shape[1]), 
      ct.c_uint32(n_clusters),
      ct.c_float(tolerance),
      ct.c_uint32(max_iterations),
      ct.c_uint32(centroid_init_type),
      centroid_init,
      ct.c_uint32(objective_function_type),

      centroids_result,
      clusters_result,
      ct.byref(converged_result))
  
    self.centroids = centroids_result
    self.clusters = clusters_result
    self.converged = (converged_result.value == 1)

    return self

  def __check_parameters(self, X, n_clusters, centroid_init, objective_function, tolerance, max_iterations):
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

    if not (objective_function == "wcss" or objective_function == "wcs"):
      raise ValueError("expected objective_function to be \"wcss\" or \"wcs\"")

    if isinstance(centroid_init, np.ndarray):
      if centroid_init.ndim != 2:
         raise TypeError("centroid_init should have 2 dimensions.")

      if centroid_init.shape[0] != n_clusters or centroid_init.shape[1] != X.shape[1]:
        raise TypeError("centroid_init should have shape: {} x {}, but has {} x {}".format(n_clusters, X.shape[1], centroid_init.shape[0], centroid_init.shape[1]))

      if centroid_init.dtype != np.float32:
        raise TypeError("centroid_init should contain elements of type np.float32.")

    else:
      if self._centroid_inits[centroid_init] is None:
        raise TypeError("invalid centroid init")


class FractalKMeans():
  """Fractal K-means clustering.

  A wrapper to a C implementation of a fractal clustering algorithm using Lloyd's algorithm. 

  Attributes
  ----------
  centroids : ndarray of shape (n_centroids, n_features) of np.float32
              Cluster centers.

  clusters  : ndarray of shape (n_layers, n_features) of np.float32
              Each row corresponds to a layers of clusters.
              Cluster label for each sample index describing which cluster the sample is in for each layer.

  converged : converged or not

  Usage
  -----
  from pynowcluster.clusters import FractalKMeans

  n = 1_000_000
  x = np.random.normal(0, 10, n)
  y = np.random.normal(0, 5, n)

  X = np.stack((x,y), axis=1)
  X = X.astype(np.float32)

  start = time.time()
  fkm = FractalKMeans().process(X)
  end = time.time()

  print(fkm.clusters)
  print(fkm.converged)

  """

  def __init__(self):
    self._centroid_inits = {
      "random" : Definitions.INIT_RANDOMLY,
      "kmeans++" : Definitions.INIT_KMEANS_PLUS_PLUS,
      "first" : Definitions.INIT_TO_FIRST_SAMPLES
    }

    self._ccore = ccore_library.get()

    self._ccore.interface_fractal_kmeans.argtypes = [
      np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), 
      ct.c_uint32, 
      ct.c_uint32,
      ct.c_uint32,
      ct.c_float,
      ct.c_uint32,
      ct.c_uint32,
      ct.c_uint32,
      ct.c_uint32,

      ct.POINTER(ct.c_uint32),
      ct.POINTER(ct.c_uint32)
    ]

    self._ccore.interface_fractal_kmeans.restype = None

    self._ccore.interface_copy_fractal_kmeans_result.argtypes = [
      ct.c_uint32, 

      np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2, flags="C_CONTIGUOUS")
    ]

    self._ccore.interface_copy_fractal_kmeans_result.restype = None

  def process(self, X, min_cluster_size = 1, centroid_init = "kmeans++", num_child_clusters = 2, objective_function = "wcss", tolerance = 0.001, max_iterations = 0):
    """
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) of np.float32
        The input samples.

    min_cluster_size : clusters with fewer or the same number of samples as min_cluster_size will stop splitting.

    centroid_init : {"kmeans++", "random", "first"}

    num_child_clusters : number of child clusters for a given cluster.

    objective_function : {"wcss", "wcs"}
                         Whether to minimize the within-cluster sum of squares (wcss) or the within-cluster sum (wcs).

    tolerance : if all centroids move less than tolerance during an update, then the centroids have converged

    max_iterations : maximum number of iterations (0 means iterate till convergens)
    """

    self.__check_parameters(X, min_cluster_size, centroid_init, num_child_clusters,objective_function, tolerance, max_iterations)

    init_type = self._centroid_inits[centroid_init]

    if objective_function == "wcss":
      objective_function_type = 1
    else:
      objective_function_type = 0

    layers_result = ct.c_uint32()
    converged_result = ct.c_uint32()
    
    self._ccore.interface_fractal_kmeans(
        X, 
        ct.c_uint32(X.shape[0]),
        ct.c_uint32(X.shape[1]),
        min_cluster_size,
        tolerance,
        max_iterations,
        init_type,
        num_child_clusters,
        ct.c_uint32(objective_function_type),

        ct.byref(layers_result),
        ct.byref(converged_result))

    self.converged = (converged_result.value == 1)

    clusters_result = np.empty((layers_result.value, X.shape[0]), dtype=np.uint32) 

    self._ccore.interface_copy_fractal_kmeans_result(
      ct.c_uint32(X.shape[0]),

      clusters_result
    )

    self.clusters = clusters_result

    return self

  def get_num_layers(self):
    return self.clusters.shape[0]

  def get_layer(self, n):
    return self.clusters[n,:]

  def __check_parameters(self, X, min_cluster_size, centroid_init, num_child_clusters, objective_function, tolerance, max_iterations):

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

    if (num_child_clusters < 2):
      raise ValueError("num_child_clusters has to be >= 2.")

    if not (objective_function == "wcss" or objective_function == "wcs"):
      raise ValueError("expected objective_function to be \"wcss\" or \"wcs\"")

    if self._centroid_inits[centroid_init] is None:
      raise TypeError("invalid centroid init")

