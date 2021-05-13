import numpy as np

def fractal_kmeans_pytorch(X):
  import time
  import torch_kmeans_main

  start = time.time()
  torch_kmeans_main.fractal_k_means_pytorch(X)
  elapsed = time.time() - start

  return elapsed

def fractal_kmeans_speedtest(D, N, f, name="*unknown", iterations=1):
  #from sklearn.datasets import make_blobs # use this import if the below doesn't work. Seems like we use different sklearn versions.
  from sklearn.datasets.samples_generator import make_blobs

  file = "fractal_kmeans_times_D{}.txt".format(D)
  
  # some of the implementations (pytorch) have some startup stuff, so lets not penalize for that
  # we execute the startup stuff here.
  f(np.array([[1,2], [2,3], [3,4]], dtype=np.float32))

  single_elapses = np.empty(iterations)
  elapses = np.empty(N.size)
  np.random.seed(0)

  append_to_file(name + "\n", file)

  for _in, n in enumerate(N):
    X, _ = make_blobs(n_samples=n, n_features=D, centers=10)
    #X = np.random.normal(0, 100, (n,D))
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


def append_to_file(str, filename):
  with open(filename, "a+") as f:
    f.write(str)


inputs = [(2,   np.array([100, 1_000, 5_000, 10_000, 100_000, 1_000_000, 2_000_000, 3_500_000, 5_000_000])),
          (4,   np.array([100, 1_000, 5_000, 10_000, 100_000, 1_000_000, 2_000_000, 3_000_000])),
          (8,   np.array([100, 1_000, 5_000, 10_000, 100_000, 1_000_000, 2_000_000])),
          (16,  np.array([100, 1_000, 5_000, 10_000, 100_000, 1_000_000])),
          (32,  np.array([100, 1_000, 5_000, 10_000, 50_000, 100_000])),
          (64,  np.array([100, 1_000, 5_000, 10_000, 50_000])),
          (128, np.array([100, 1_000, 5_000, 10_000]))
        ]

tests_count = len(inputs)

for i, (D, N) in enumerate(inputs):
  fractal_kmeans_speedtest(D, N, fractal_kmeans_pytorch, "pytorch")
  print("D={} N={} done, {:.2f}%".format(D, N[-1], (i+1.0) / tests_count * 100.))
