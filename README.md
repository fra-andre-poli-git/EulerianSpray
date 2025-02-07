# EulerianSpray
This is the repository of my APSC/PACS project on Eulerian Sprays.

The licensing is in according to what I read on deal.ii readme:
https://www.dealii.org/9.5.0/readme.html#license

For the moment I am using the version installed from apt-get:
https://github.com/dealii/dealii/wiki/Debian-and-Ubuntu
Probably I will use the installation of deal.ii in the mk modules:
https://github.com/pcafrica/mk

To run it

mkdir build
cd build
cmake ..
make
mkdir results
./main ../input/INPUTFILENAME.prm

where INPUTFILENAME.prm is the desired input file