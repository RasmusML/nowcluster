from ctypes import cdll
import sys
import os  

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
                    
            def get_library_path():
                package = __package__.split(".")[-1]
                lib_absolute_path = __file__.split(package, 1)[0]
                ccore_relative_path = "ccore" + os.path.sep + "build" + os.path.sep + "nowcluster" + get_os_extension()
                return lib_absolute_path + ccore_relative_path
                
            full_path = get_library_path()
            ccore_library.lib = cdll.LoadLibrary(full_path)
        
        return ccore_library.lib



