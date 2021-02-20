if not exist "build" mkdir "build"
pushd build
cl ..\src\*.c /LD /link -EXPORT:kmeans
del *.lib
del *.obj
del *.exp
popd
