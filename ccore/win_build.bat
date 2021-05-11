if not exist "build" mkdir "build"
pushd build
cl ..\*.c /LD /O2 /openmp /link -EXPORT:interface_kmeans -EXPORT:interface_fractal_kmeans -EXPORT:interface_copy_fractal_kmeans_result /OUT:nowcluster.dll
popd
