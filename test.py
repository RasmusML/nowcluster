import pynowcluster.clusters
import numpy as np
import time

def kmeans_comparison_test():
  import sklearn.cluster

  np.random.seed(1)

  n = 7_000_0000
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
  nc_result = pynowcluster.clusters.KMeans().process(X, clusters, centroid_init, "wcss", max_iterations=300)  #sklearn uses max 300 iterations
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
  
#fractal_k_means_test()
#fractal_k_means_test2()
#fractal_k_means_test3()

def fractal_k_means_test():
  np.random.seed(0)
  n = 10_000_000
  x = np.random.normal(0, 100, n)
  y = np.random.normal(0, 50, n)
  
  X = np.stack((x,y), axis=1)
  X = X.astype(np.float32)

  start = time.time()
  fkm = pynowcluster.clusters.FractalKMeans().process(X, objective_function="wcs", max_iterations=300) # the pytorch implementation iterates max 300 times, so lets do the same.
  end = time.time()

  print("nowcluster took:", end - start)
  print("nowcluster clusters:\n", fkm.clusters)
  print("nowcluster layers:\n", fkm.clusters.shape)  
  print("nowcluster converged:\n", fkm.converged)



### Fractal K-means tests ###

def fractal_kmeans_pynowcluster(X):
  import pynowcluster.clusters
  import time

  fractal_kmeans = pynowcluster.clusters.FractalKMeans()

  start = time.time()
  fractal_kmeans.process(X, objective_function="wcs")
  elapsed = time.time() - start

  return elapsed


def fractal_kmeans_pytorch(X):
  import time
  import torch_kmeans_main

  start = time.time()
  torch_kmeans_main.fractal_k_means_pytorch(X)
  elapsed = time.time() - start

  return elapsed

def fractal_kmeans_speedtest(D, N, f, name="*unknown", iterations=1):
  from sklearn.datasets import make_blobs

  file = "fractal_kmeans_times_D{}.txt".format(D)
  
  # some of the implementation have some startup stuff, so lets not penealize for that
  # we execute the startup stuff here.
  f(np.array([[1,2], [2,3], [3,4]], dtype=np.float32))

  single_elapses = np.empty(iterations)
  elapses = np.empty(N.size)
  np.random.seed(0)

  append_to_file(name + "\n", file)

  for _in, n in enumerate(N):
    #X, _ = make_blobs(n_samples=n, n_features=D, centers=10)
    X = np.random.normal(0, 100, (n,D))
    X = X.astype(dtype=np.float32)

    for i in range(iterations):
      single_elapses[i] = f(X)
    
    avg_elapsed = single_elapses.mean()
    elapses[_in] = avg_elapsed

    print("D:{} N:{} {:.2f}s".format(D, n, avg_elapsed))

    output = "{} {:.2f}\n".format(n, avg_elapsed)
    append_to_file(output, file)
    
  append_to_file("\n", file)

  return elapses


### K-means tests ###

def kmeans_pyclustering(X, n_clusters):
  from pyclustering.cluster.kmeans import kmeans
  from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
  from pyclustering.core.wrapper import ccore_library
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
  import time

  start = time.time()
  nc_result = pynowcluster.clusters.KMeans().process(X, n_clusters, centroid_init="kmeans++", max_iterations=100)
  elapsed = time.time() - start

  return elapsed

def kmeans_sklearn(X, n_clusters):
  import sklearn.cluster
  import time

  start = time.time()
  sk_result = sklearn.cluster.KMeans(n_clusters=n_clusters, init="k-means++", max_iter=100, n_init=1, algorithm="full").fit(X)
  elapsed = time.time() - start

  return elapsed

def k_means_speed_test(K, D, N, f, name="*unknown", iterations=1):
  from sklearn.datasets import make_blobs

  file = "kmeans_times_K{}_D{}.txt".format(K, D)

  single_elapses = np.empty(iterations)
  elapses = np.empty(N.size)
  np.random.seed(0)

  append_to_file(name + "\n", file)

  for _in, n in enumerate(N):
    X, _ = make_blobs(n_samples=n, n_features=D, centers=10)
    #X = np.random.normal(0, 100, (n,D))
    X = X.astype(dtype=np.float32)

    for i in range(iterations):
      single_elapses[i] = f(X, K)
    
    avg_elapsed = single_elapses.mean()
    elapses[_in] = avg_elapsed
    
    print("K:{} D:{} N:{} {:.2f}s".format(K, D, n, avg_elapsed))

    output = "{} {:.2f}\n".format(n, avg_elapsed)
    append_to_file(output, file)
  
  append_to_file("\n", file)

  return elapses

def append_to_file(str, filename):
  with open(filename, "a+") as f:
    f.write(str)

### Plotting ###

def load_test_runs(file):
  import os
  import re

  f = open(file, "r")
  content = f.read()

  content = content.rstrip().split(os.linesep)

  files = []

  name = "*unknown"
  Ns = []
  times = []

  name_next = True

  for line in content:
    if line == "":
      files.append((name, Ns, times))

      Ns = []
      times = []
      name = "*unknown"
      name_next = True

    elif name_next:
      name_next = False
      name = line

    else:
      split = line.split(" ")
      N = int(split[0])
      time = float(split[1])

      Ns.append(N)
      times.append(time)

  files.append((name, Ns, times))

  return files

def plot_kmeans(K, D, tests, save=True):
  import matplotlib.pyplot as plt

  title = "{}-means D={}".format(K, D)
  plt.title(title)
  plt.grid()

  plt.xlabel("N")
  plt.ylabel("execution time (s)")
  
  for (name, Ns, times) in tests:
    plt.plot(Ns, times, label=name)
    plt.scatter(Ns, times)
    
  plt.legend()

  if save:
    plt.savefig("kmeans_D{}_K{}.png".format(D, K))

  plt.show()

def plot_fractal_kmeans(D, ns1, ns2, ns3, times1, times2, times3, save=True):
  import matplotlib.pyplot as plt

  title = "Fractal K-means D={}".format(D)
  plt.title(title)

  plt.xlabel("N")
  plt.ylabel("execution time (s)")
  
  plt.plot(ns1, times1, 'b', label="NowCluster")
  plt.scatter(ns1, times1, color='b')
  
  plt.plot(ns2, times2, 'g', label="PyTorch (1060)")
  plt.scatter(ns2, times2, color='g')
  
  plt.plot(ns3, times3, 'r', label="PyTorch (GTX 2080)")
  plt.scatter(ns3, times3, color='r')

  plt.legend()

  if save:
    plt.savefig("fractal_kmeans_D{}.png".format(D))

  plt.show()


#kmeans_comparison_test()

# K-means
K = 80
D = 20

"""
N = np.array([1_000, 5_000, 10_000, 100_000])
elapses_pyclustering = k_means_speed_test(K, D, N, kmeans_pyclustering, "pyclustering")
print("kmeans pyclustering done")


N = np.array([1_000, 5_000, 10_000, 100_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000, 8_000_000, 10_000_000, 14_000_000, 18_000_000])
elapses_sklearn = k_means_speed_test(K, D, N, kmeans_sklearn, "sklearn")
print("kmeans sklearn done")


N = np.array([1_000, 5_000, 10_000, 100_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000, 8_000_000, 10_000_000, 14_000_000, 18_000_000])
elapses_nowcluster = k_means_speed_test(K, D, N, kmeans_pynowcluster, "nowcluster")
print("kmeans pynowcluster done")

tests = load_test_runs("kmeans_times_K{}_D{}.txt".format(K, D))

plot_kmeans(K=K, D=D, tests=tests)
"""


# Fractal K-means
"""
inputs = [#(2,   np.array([100, 1_000, 5_000, 10_000, 100_000, 1_000_000, 2_000_000, 3_500_000, 5_000_000])),
          #(4,   np.array([100, 1_000, 5_000, 10_000, 100_000, 1_000_000, 2_000_000, 3_000_000])),
          #(8,   np.array([100, 1_000, 5_000, 10_000, 100_000, 1_000_000, 2_000_000])),
          #(16,  np.array([100, 1_000, 5_000, 10_000, 100_000, 1_000_000])),
          #(32,  np.array([100, 1_000, 5_000, 10_000, 50_000, 100_000])),
          (64,  np.array([100, 1_000, 5_000, 10_000, 50_000])),
          (128, np.array([100, 1_000, 5_000, 10_000]))
        ]

tests_count = len(inputs)

for i, (D, N) in enumerate(inputs):
  fractal_kmeans_speedtest(D, N, fractal_kmeans_pynowcluster, "nowcluster")
  print("D={} N={} done, {:.2f}%".format(D, N[-1], (i+1.0) / tests_count * 100.))

for i, (D, N) in enumerate(inputs):
  fractal_kmeans_speedtest(D, N, fractal_kmeans_pytorch, "pytorch")
  print("D={} N={} done, {:.2f}%".format(D, N[-1], (i+1.0) / tests_count * 100.))

"""
D = 16

(_, ns1, times1), (_, ns2, times2), (_, ns3, times3) = load_test_runs("fractal_kmeans_times_D{}.txt".format(D))


plot_fractal_kmeans(D, ns1, ns2, ns3, times1, times2, times3)