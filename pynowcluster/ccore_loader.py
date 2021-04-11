from ctypes import cdll
from sys import platform as _platform
import os
import pynowcluster


if (_platform == "win32"):
    PATH_NOWCLUSTER_CCORE = pynowcluster.__path__[0] + os.sep + "ccore" + os.sep + "build" + os.sep + "nowcluster.dll"
elif (_platform == "linux"):
    PATH_NOWCLUSTER_CCORE = pynowcluster.__path__[0] + os.sep + "ccore" + os.sep + "build" + os.sep + "nowcluster.so"
else:
    raise OSError(f"{_platform} unsupported platform." )

class ccore_library:
    
    lib = None

    @staticmethod
    def get():           

        if ccore_library.lib is None:     
            ccore_library.lib = cdll.LoadLibrary(PATH_NOWCLUSTER_CCORE)
        
        return ccore_library.lib



