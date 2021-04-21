import pynowcluster.clusters
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import time


def kmeans_test():
  np.random.seed(1)

  n = 10
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
  nc_result = pynowcluster.clusters.KMeans().process(X, clusters, centroid_init)
  end = time.time()

  print("nowcluster took:", end - start)
  print("nowcluster centroids:\n", nc_result._centroids)
  #print("nowcluster groups:", groups)

def fractal_k_means_test():
  np.random.seed(0)
  n = 10
  x = np.random.normal(0, 10, n)
  y = np.random.normal(0, 5, n)
  
  X = np.stack((x,y), axis=1)
  X = X.astype(np.float32)

  start = time.time()
  fkm = pynowcluster.clusters.FractalKMeans().process(X)
  end = time.time()

  print("nowcluster took:", end - start)
  print("nowcluster clusters:\n", fkm._clusters)
  print("nowcluster layers:\n", fkm._clusters.shape)

#kmeans_test()
fractal_k_means_test()