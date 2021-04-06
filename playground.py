from pynowcluster.kmeans import k_means, fractal_k_means
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def kmeans_test():
  n = 1000
  x = np.random.normal(0, 10, n);
  y = np.random.normal(0, 5, n);

  X = np.stack((x,y), axis=1)
  X = X.astype(np.float32);

  clusters = 10

  centroids, groups = k_means(X, clusters)

  for i in range(clusters):
      points = X[groups == i] 
      plt.scatter(points[:,0], points[:,1])

  plt.scatter(centroids[:,0], centroids[:,1], color="black")
  plt.title(f"{clusters}-means (n={n})")
  plt.show()
      


def fractal_k_means_test():
  n = 24
  x = np.random.normal(0, 10, n);
  y = np.random.normal(0, 5, n);
  

  X = np.stack((x,y), axis=1)
  X = X.astype(np.float32);

  layers = 7

  groups = fractal_k_means(X, layers)
  print(groups)

# fractal_k_means_test()


def k_means_sci_test():
  n = 2400
  x = np.random.normal(0, 10, n);
  y = np.random.normal(0, 5, n);
  

  X = np.stack((x,y), axis=1)
  X = X.astype(np.float32);

  kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
  print("done")
  
fractal_k_means_test();
#k_means_sci_test();
