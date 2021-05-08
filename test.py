import pynowcluster.clusters
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import time


def kmeans_test():
  np.random.seed(1)

  n = 7_000_000
  x = np.random.normal(0, 10, n)
  y = np.random.normal(0, 5, n)

  X = np.stack((x,y), axis=1)
  X = X.astype(np.float32)

  clusters = 10

  centroid_init = X[:clusters,]
  print(centroid_init.shape)
  #centroid_init = np.zeros((clusters, 2), dtype=np.float32)
  print("centroid init:\n", centroid_init)
  """
  start = time.time()
  sk_result = sklearn.cluster.KMeans(n_clusters=clusters, init=centroid_init, n_init=1, algorithm="full").fit(X)
  end = time.time()
  print("sklearn took:", end - start)
  print("sklearn centroids:\n", sk_result.cluster_centers_)
  print("sklearn groups:", sk_result.labels_)
  """
  
  start = time.time()
  nc_result = pynowcluster.clusters.KMeans().process(X, clusters, centroid_init, "wcs", 0.001)
  end = time.time()

  print("nowcluster took:", end - start)
  print("nowcluster centroids:\n", nc_result.centroids)
  print(nc_result.converged)
  #print("nowcluster groups:", groups)

def fractal_k_means_test():
  np.random.seed(0)
  n = 7_000_000
  x = np.random.normal(0, 10, n)
  y = np.random.normal(0, 5, n)
  
  X = np.stack((x,y), axis=1)
  X = X.astype(np.float32)

  start = time.time()
  fkm = pynowcluster.clusters.FractalKMeans().process(X)
  end = time.time()

  print("nowcluster took:", end - start)
  print("nowcluster clusters:\n", fkm.clusters)
  print("nowcluster layers:\n", fkm.clusters.shape)  
  print("nowcluster converged:\n", fkm.converged)
  
#kmeans_test()
#fractal_k_means_test()

def test1():
  np.random.seed(0)
  n = 700
  x = np.random.normal(0, 10, n)
  y = np.random.normal(0, 5, n)
    
  X = np.stack((x,y), axis=1)
  X = X.astype(np.float32)

  fractalKMeans = pynowcluster.clusters.FractalKMeans().process(X, min_cluster_size=10, objective_function="wcs", tolerance=1)
  #Kmeans = pynowcluster.clusters.KMeans().process(X, 2, objective_function="wcs", tolerance=0.001)

  print("done")

def test2():
  np.random.seed(0)

  data_size = 100
  num_iters = 50
  num_clusters = 4

  # sample from Gaussians 
  data1 = np.random.normal((5,5), (4, 4), (data_size,2))
  data2 = np.random.normal((4,60), (3,3), (data_size, 2))
  data3 = np.random.normal((65, 82), (5, 5), (data_size,2))
  data4 = np.random.normal((60, 4), (5, 5), (data_size,2))

  # Combine the data to create the final dataset
  X = np.concatenate((data1,data2, data3, data4), axis = 0)
  X = X.astype(np.float32)

  np.random.shuffle(X)

  kMeans = pynowcluster.clusters.KMeans().process(X, num_clusters, objective_function="wcss")
  print("converged:", kMeans.converged)

#test2()
fractal_k_means_test()