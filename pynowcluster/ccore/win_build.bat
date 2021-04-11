if not exist "build" mkdir "build"
pushd build
cl ..\*.cpp /LD /O2 /link -EXPORT:k_means -EXPORT:fractal_k_means /OUT:nowcluster.dll
popd
