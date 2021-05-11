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

n = 7_000_00
x = np.random.normal(0, 10, n)
y = np.random.normal(0, 5, n)

X = np.stack((x,y), axis=1)
X = X.astype(np.float32)

clusters = 10

# Load list of points for cluster analysis.
sample = X

# Prepare initial centers using K-Means++ method.
initial_centers = kmeans_plusplus_initializer(sample, clusters).initialize()
# Create instance of K-Means algorithm with prepared centers.
kmeans_instance = kmeans(sample, initial_centers)

start = time.time()
# Run cluster analysis and obtain results.
kmeans_instance.process()

end = time.time()

print("pyclustering took:", end - start)

clusters = kmeans_instance.get_clusters()
final_centers = kmeans_instance.get_centers()
