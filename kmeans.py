from ctypes import *
import numpy as np

kmeans_lib = cdll.LoadLibrary("./bin/kmeans.so")
kmeans = kmeans_lib.kmeans
kmeans.argtypes = [c_int32, c_uint32]
kmeans.restype = None

kmeans(c_int32(-10), c_uint32(12))
