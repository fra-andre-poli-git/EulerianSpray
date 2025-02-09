# EulerianSpray
This is the repository of my APSC/PACS project on Eulerian Sprays.

The licensing is in according to what I read on deal.ii readme:
https://www.dealii.org/9.5.0/readme.html#license

The code is developed under version 9.5.1 installed from apt-get:
https://github.com/dealii/dealii/wiki/Debian-and-Ubuntu
But then it was made to work also with the installation of deal.ii in the mk
modules:
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