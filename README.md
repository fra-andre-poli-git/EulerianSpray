# EulerianSpray
This is the repository of my APSC/PACS project on Eulerian Sprays.

The license is the same as that of the library, as evinced in deal.II README:
https://www.dealii.org/9.5.0/readme.html#license

The code is developed under version 9.5.1 installed from apt-get:
https://github.com/dealii/dealii/wiki/Debian-and-Ubuntu

But then CMakeLists.txt was modified to work also (and preferentially) with the 
installation of deal.ii in the mk modules:
https://github.com/pcafrica/mk

The structure of the classes is taken from tutorial 67 of deal.II library, which
solves Euler equations. The commented code is found at:
https://www.dealii.org/current/doxygen/deal.II/step_67.html

Most of the parameters are set at run time via input file, that can be found in
./input directory. The physical dimension of the problem is set at compile time,
in the main.cpp file, as well as the polynomial degree.

To run the code you need to follow the steps below.

```bash
mkdir build

cd build

cmake ..

make release

make

mkdir results

./main ../input/INPUTFILENAME.prm
```
where INPUTFILENAME.prm is the desired input file.
I underline the switch to release mode, since otherwise 2d simulation would be
unfeasible. To check the build type:
```bash
grep CMAKE_BUILD_TYPE CMakeCache.txt
```