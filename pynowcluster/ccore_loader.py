from ctypes import cdll
from sys import platform as _platform
import os
import platform
import pynowcluster

core_architecture = None

if platform.architecture()[0] == "64bit":
    core_architecture = "64-bit"
else:
    core_architecture = "32-bit"

if (_platform == "win32"):
    PATH_NOWCLUSTER_CCORE = pynowcluster.__path__[0] + os.sep + "ccore" + os.sep + core_architecture + os.sep + "win32" + os.sep + "nowcluster.dll"
elif (_platform == "linux"):
    PATH_NOWCLUSTER_CCORE = pynowcluster.__path__[0] + os.sep + "ccore" + os.sep + core_architecture + os.sep + "linux" + os.sep + "nowcluster.so"
else:
    raise OSError(f"{_platform} unsupported platform." )

class ccore_library:
    
    __library = None
    __initialized = False

    @staticmethod
    def get():           

        if not ccore_library.__initialized:

            if os.path.exists(PATH_NOWCLUSTER_CCORE) is False:
                raise FileNotFoundError("A dynamic library for your platform {} {} does not exist. Please build the lib manually.".format(core_architecture, _platform))
                     
            ccore_library.__library = cdll.LoadLibrary(PATH_NOWCLUSTER_CCORE)
            ccore_library.__initialized = True

        return ccore_library.__library



