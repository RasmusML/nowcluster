from ctypes import *


kmeans_lib = cdll.LoadLibrary("./bin/kmeans.so")
kmeans_lib.go()
kmeans_lib.hello()
