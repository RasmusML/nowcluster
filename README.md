# nowcluster
nowcluster is a small C clustering library with Python bindings.

## Usage
```
import numpy as np
from pynowcluster.clusters import FractalKMeans

n = 1_000_000
x = np.random.normal(0, 10, n)
y = np.random.normal(0, 5, n)

X = np.stack((x,y), axis=1)
X = X.astype(np.float32)

fkm = FractalKMeans().process(X)

print(fkm.clusters)
print(fkm.converged)
```
## Installation
### Windows
1. Update win_startup.bat to your vcvarsall.bat
2. Run `win_startup.bat`
3. Run `win_build.bat`
4. Move the DLL into the appropriate pynowclusters folder based on target machine (x32/x64) and os (win).


### Linux
1. Run the makefile `make` located inside the folder "ccore" to create the shared library. 
2. Move the shared library into the appropriate pynowclusters folder based on target machine (x32/x64) and os (linux).
