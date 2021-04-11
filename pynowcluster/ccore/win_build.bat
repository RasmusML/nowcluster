if not exist "build" mkdir "build"
pushd build
cl ..\*.cpp /LD /O2 /link -EXPORT:k_means -EXPORT:fractal_k_means -EXPORT:copy_fractal_k_means_result /OUT:nowcluster.dll
popd
