import numpy as np 
import matplotlib.pyplot as plt 
import random
import time 

# Size of dataset to be generated. The final size is 4 * data_size
data_size = 10_000_000
num_iters = 19
num_clusters = 4

np.random.seed(1)

x = np.random.normal(0, 10, data_size)
y = np.random.normal(0, 5, data_size)

data = np.stack((x,y), axis=1)
data = data.astype(np.float32)

# Combine the data to create the final dataset
#data = np.concatenate((data1,data2, data3, data4), axis = 0)


# Set random seed for reproducibility 

# Initialise centroids
centroids = data[:num_clusters,]

print(centroids)

# Create a list to store which centroid is assigned to each dataset
assigned_centroids = np.zeros(len(data), dtype = np.int32)

def compute_l2_distance(x, centroid):
    # Compute the difference, following by raising to power 2 and summing
    dist = ((x - centroid) ** 2).sum(axis = x.ndim - 1)
    
    return dist

def get_closest_centroid(x, centroids):
    
    # Loop over each centroid and compute the distance from data point.
    dist = compute_l2_distance(x, centroids)

    # Get the index of the centroid with the smallest distance to the data point 
    closest_centroid_index =  np.argmin(dist, axis = 1)
    
    return closest_centroid_index

def compute_sse(data, centroids, assigned_centroids):
    # Initialise SSE 
    sse = 0

    # Compute SSE
    sse = compute_l2_distance(data, centroids[assigned_centroids]).sum() / len(data)
    
    return sse

# Number of dimensions in centroid
num_centroid_dims = data.shape[1]

# List to store SSE for each iteration 
sse_list = []

# Start time
tic = time.time()

# Main Loop
for n in range(num_iters):
    # Get closest centroids to each data point
    assigned_centroids = get_closest_centroid(data[:, None, :], centroids[None,:, :])    

    # Compute new centroids
    for c in range(centroids.shape[1]):
        # Get data points belonging to each cluster 
        cluster_members = data[assigned_centroids == c]
        
        # Compute the mean of the clusters
        cluster_members = cluster_members.mean(axis = 0)
        
        # Update the centroids
        centroids[c] = cluster_members
    
    # Compute SSE
    #sse = compute_sse(data.squeeze(), centroids.squeeze(), assigned_centroids)
    #sse_list.append(sse)


# End time
toc = time.time()


print(round(toc - tic, 4))