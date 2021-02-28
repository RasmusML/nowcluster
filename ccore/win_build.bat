if not exist "build" mkdir "build"
pushd build
cl ..\*.cpp /LD /link -EXPORT:k_means /OUT:nowcluster.dll
del *.exp
del *.lib
del *.obj
popd
