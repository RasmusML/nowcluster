from ctypes import cdll
import sys
import os
import pynowcluster
import platform
  
class ccore_library:
    
    lib = None

    @staticmethod
    def get():
        if ccore_library.lib is None:
        
            def get_os_extension():
                if sys.platform.startswith("win32"):
                    return ".dll"
                elif sys.platform.startswith("linux"):
                    return ".so"
                else:
                    raise OSError(f"{os} unsupported platform.")
                    
            def get_library_path(): #fix
                dir = pynowcluster.__path__[0]
                return dir[:-len(pynowcluster.__name__)] + "ccore/build/nowcluster" + get_os_extension()
                    
            full_path = get_library_path()
            ccore_library.lib = cdll.LoadLibrary(full_path)
        
        return ccore_library.lib



