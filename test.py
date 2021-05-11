import pynowcluster.clusters
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import time

def kmeans_comparison_test():
  np.random.seed(1)

  n = 7_000_00
  x = np.random.normal(0, 10, n)
  y = np.random.normal(0, 5, n)

  X = np.stack((x,y), axis=1)
  X = X.astype(np.float32)

  clusters = 10

  centroid_init = X[:clusters,]
  #centroid_init = np.zeros((clusters, 2), dtype=np.float32)
  print("centroid init:\n", centroid_init)
  
  start = time.time()
  sk_result = sklearn.cluster.KMeans(n_clusters=clusters, init=centroid_init, n_init=1, algorithm="full").fit(X)
  end = time.time()
  print("sklearn took:", end - start)
  print("sklearn centroids:\n", sk_result.cluster_centers_)
  print("sklearn groups:", sk_result.labels_)
  
  
  start = time.time()
  nc_result = pynowcluster.clusters.KMeans().process(X, clusters, centroid_init, "wcss")
  end = time.time()

  print("nowcluster took:", end - start)
  print("nowcluster centroids:\n", nc_result.centroids)
  print(nc_result.converged)
  #print("nowcluster groups:", groups)


def fractal_k_means_test3():
  np.random.seed(0)
  
  X = np.array([[18.675579, 11.348773], [14.940791, 6.151453], [14.542735, 1.890813]], dtype=np.float32)
  centroid_inits = np.array([[14.940791, 6.151453],[18.675579, 11.348773]], dtype=np.float32)
  print(X)
  print(centroid_inits)

  start = time.time()
  fkm = pynowcluster.clusters.KMeans().process(X, 2, centroid_init=centroid_inits,objective_function="wcs")
  end = time.time()

  print("nowcluster took:", end - start)
  print("nowcluster clusters:\n", fkm.clusters)
  print("nowcluster layers:\n", fkm.clusters.shape)  
  print("nowcluster converged:\n", fkm.converged)
  print("nowcluster centroids:\n", fkm.centroids)

def fractal_k_means_test2():
  np.random.seed(0)

  X = np.array([[18.675579, 11.348773], [14.940791, 6.151453], [9.787380, 4.322181], [14.542735, 1.890813], [9.500884, 0.228793]], dtype=np.float32)

  start = time.time()
  fkm = pynowcluster.clusters.FractalKMeans().process(X, objective_function="wcs")
  end = time.time()

  print("nowcluster took:", end - start)
  print("nowcluster clusters:\n", fkm.clusters)
  print("nowcluster layers:\n", fkm.clusters.shape)  
  print("nowcluster converged:\n", fkm.converged)
  
def fractal_k_means_test():
  np.random.seed(0)
  n = 1_000_000
  x = np.random.normal(0, 100, n)
  y = np.random.normal(0, 50, n)
  
  X = np.stack((x,y), axis=1)
  X = X.astype(np.float32)

  start = time.time()
  fkm = pynowcluster.clusters.FractalKMeans().process(X, objective_function="wcs")
  end = time.time()

  print("nowcluster took:", end - start)
  print("nowcluster clusters:\n", fkm.clusters)
  print("nowcluster layers:\n", fkm.clusters.shape)  
  print("nowcluster converged:\n", fkm.converged)


def fractal_k_means_performance_test(N, D, iterations, f):
  np.random.seed(0)

  X = np.random.normal(0, 100, (N,D))
  X = X.astype(dtype=np.float32)

  elapses = np.empty(iterations)

  for i in range(iterations):
    start = time.time()
    f(X)  
    elapsed = time.time() - start
    elapses[i] = elapsed
  
  return np.mean(elapses)

def comprehensive_fractal_k_means_performanc_test(f):
  sizes = np.array([1000, 10_000, 100_000, 1_000_000, 10_000_000])
  dims = np.array([2, 4, 8, 16, 32])
  iterations = 2

  elapses = np.empty((sizes.size, dims.size))

  for d, dim in enumerate(dims):
    for s, size in enumerate(sizes):
      elapsed = fractal_k_means_performance_test(size, dim, iterations, f)
      elapses[s,d] = elapsed

      print("{:.2f}s N={} D={}".format(elapsed, size, dim))


def f_fkm_wrapper(fkm, min_cluster_size = 10, objective_function = "wcs"):
  return lambda X : fkm.process(X, min_cluster_size=min_cluster_size, objective_function=objective_function)

"""
fractal_k_means_test()
fractal_k_means_test2()
fractal_k_means_test3()
"""

fractal_k_means = pynowcluster.clusters.FractalKMeans()
f = f_fkm_wrapper(fractal_k_means)

comprehensive_fractal_k_means_performanc_test(f)

#kmeans_comparison_test()