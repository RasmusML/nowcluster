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

#fractal_k_means = pynowcluster.clusters.FractalKMeans()
#f = f_fkm_wrapper(fractal_k_means)

#comprehensive_fractal_k_means_performanc_test(f)

#kmeans_comparison_test()

def kmeans_pyclustering(X, n_clusters):
  from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
  from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
  from pyclustering.samples.definitions import FCPS_SAMPLES
  from pyclustering.utils import read_sample
  from pyclustering.core.wrapper import ccore_library

  import numpy as np
  import time

  # make sure the C/C++ implementation is actually available.
  ccore = ccore_library.workable()
  assert ccore

  initial_centers = kmeans_plusplus_initializer(X, n_clusters).initialize()
  kmeans_instance = kmeans(X, initial_centers)

  start = time.time()
  kmeans_instance.process()
  elapsed = time.time() - start

  return elapsed


def kmeans_pynowcluster(X, n_clusters):
  import pynowcluster.clusters
  import numpy as np

  import time

  start = time.time()
  nc_result = pynowcluster.clusters.KMeans().process(X, n_clusters, centroid_init="kmeans++")

  elapsed = time.time() - start

  return elapsed


def append_to_file(str, filename):
  with open(filename, "a") as f:
    f.write(str)

def k_means_speed_test():

  kmeans_times_file = "kmeans_times.txt"

  K = np.array([2, 4, 6, 8, 10, 20, 40, 100])
  D = np.array([2, 4, 6, 8, 10, 20, 40])
  N = np.array([10, 100, 1_000, 5_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000, 10_000_000])

  iterations = 2

  single_elapses = np.empty(iterations)

  elapses = np.empty((2, K.size, D.size, N.size))
  np.random.seed(0)

  for ik, k in enumerate(K):
    for id, d in enumerate(D):
      for _in, n in enumerate(N):

        X = np.random.normal(0, 100, (n,d))
        X = X.astype(dtype=np.float32)

        for i in range(iterations):
          single_elapses[i] = kmeans_pyclustering(X, k)
        
        avg_elapsed1 = single_elapses.mean()
        elapses[0,ik,id,_in] = avg_elapsed1
        
        for i in range(iterations):
          single_elapses[i] = kmeans_pynowcluster(X, k)
        
        avg_elapsed2 = single_elapses.mean()
        elapses[1,ik,id,_in] = avg_elapsed2

        summary = "K:{} D:{} N:{} {:.2f}s <-> {:.2f}s".format(k, d, n, avg_elapsed1, avg_elapsed2)
        append_to_file(summary + "\n", kmeans_times_file)
        
        print(summary)

#k_means_speed_test()

def plot():
  import matplotlib.pyplot as plt

  times1 = np.arange(1, 100)
  times2 = np.arange(1, 100) - 10

  print(times1.shape)

  steps = np.arange(1, times1.shape[0] + 1)

  title = "{}-means D={}".format(2, 4)
  plt.title(title)
  
  plt.plot(steps, times1, 'b', label="PyClustering")
  plt.scatter(steps, times1)
  
  plt.plot(steps, times2, 'g', label="NowCluster")
  plt.scatter(steps, times2)
  
  plt.legend()
  plt.show()

plot()