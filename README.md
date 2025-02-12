# EulerianSpray
This is the repository of my APSC/PACS project on Eulerian Sprays.

The license is the same as that of the library, as evinced in deal.ii README:

https://www.dealii.org/9.5.0/readme.html#license

The code is developed under version 9.5.1 installed from apt-get:

https://github.com/dealii/dealii/wiki/Debian-and-Ubuntu

But then CMakeLists.txt was modified to work also (and preferentially) with the 
installation of deal.ii in the mk modules:

https://github.com/pcafrica/mk

Most of the parameters are set at run time via input file, that can be found in
./input directory. The physical dimension of the problem is set at compile time,
in the main.cpp file, as well as the polynomial degree.

To run the code you need to follow the steps below

mkdir build

cd build

cmake ..

make

mkdir results

./main ../input/INPUTFILENAME.prm

where INPUTFILENAME.prm is the desired input file
