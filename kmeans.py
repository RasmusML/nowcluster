import sys
from ctypes import *
import numpy as np

def load_library():
    global kmeans_lib
    os = sys.platform

    if os == "linux":
        kmeans_lib = cdll.LoadLibrary("./bin/kmeans.so")
    else:
        print(f"{os} unsupported platform")
        assert False

def init_library():
    global kmeans
    kmeans = kmeans_lib.kmeans
    kmeans.argtypes = [c_int32, c_uint32]
    kmeans.restype = None


load_library()
init_library()

kmeans(c_int32(-10), c_uint32(12))
