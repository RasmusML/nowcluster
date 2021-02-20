if not exist "build" mkdir "build"
pushd build
cl ..\src\*.c /LD /link -EXPORT:k_means
del *.exp
del *.lib
del *.obj
popd
